import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Embedding Model & Existing ChromaDB
embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "device": device,
        "model_kwargs": {"torch_dtype": torch.float16 if device == "cuda" else torch.bfloat16}
    },
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    query_encode_kwargs={
        "prompt": "Instruct: Given a cooking question or dish request, retrieve relevant recipe passages with ingredients and instructions.\nQuery: ",
        "normalize_embeddings": True
    }
)

vectorstore = Chroma(
    persist_directory="./chroma_rag_db",
    embedding_function=embeddings
)

# 2. Load Qwen2.5-3B-Instruct (4-bit on GPU, bfloat16 on free CPU Space)
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

if device == "cuda":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

stop_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|im_end|>")
]

hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=800,
    do_sample=False,
    repetition_penalty=1.1,
    eos_token_id=stop_ids,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=hf_pipe))

# 3. Build RetrievalQA Chain
recipe_template = """You are an expert chef and helpful recipe assistant.
Answer the user's question using ONLY the recipe context provided below.
Do not invent ingredients or steps that are not in the context.
If the requested recipe is not in the context, reply: "I couldn't find that recipe in your uploaded cookbooks."

When providing a recipe, format your response clearly as:
- **Recipe Name**
- **Prep / Cook Time** (if mentioned)
- **Ingredients** (bulleted list)
- **Instructions** (numbered steps)

Recipe Context:
{context}

User Question: {question}"""

QA_PROMPT = PromptTemplate(template=recipe_template, input_variables=["context", "question"])

# Used when search_type="mmr" (Maximal Marginal Relevance) is turned on. 
# fetch_k: 15 get top 15 chunks that are closest to your question
# k: 6 look through those 15 candidates and picks 6 chunks that are relevant 
# to question and different from each other to send Qwen2.5-3B to gen answer.

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# 4. Gradio Interface Function
def ask_cookbook(question):
    if not question.strip():
        return "Please enter a recipe question."
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    
    sources = set()
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "Cookbook").split("/")[-1]
        page = doc.metadata.get("page", "N/A")
        sources.add(f"- {src} (Page {page})")
    
    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join(sorted(sources))
    return answer

demo = gr.Interface(
    fn=ask_cookbook,
    inputs=gr.Textbox(lines=2, placeholder="e.g., How do I make garlic butter bread?"),
    outputs=gr.Markdown(),
    title="Cookbook Recipe Assistant",
    description="Ask questions about recipes from your uploaded cookbooks."
)

if __name__ == "__main__":
    demo.launch()
