import os
from llama_parse import LlamaParse

DOCUMENTS_DIR = r"C:\Users\bharg\OneDrive\Desktop\loan prediction via sml\llama index\user_documents"

# =====================================================
# INITIALIZE LLAMAPARSE
# =====================================================
parser = LlamaParse(
    api_key="llx-u8dkLUemOODMi90ETG345s7RK8dGHuP5mohmrzkzECnf0r3v",
    result_type="text",
    language="en",
    verbose=True,
    num_workers=4,
)

# =====================================================
# EXTRACT TEXT FROM ALL FILES
# =====================================================
def extract_all_documents_text(folder_path):
    extracted_texts = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        if not filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            print(f"Skipping unsupported file: {filename}")
            continue

        print(f"\n📄 Parsing: {filename}")

        try:
            documents = parser.load_data(file_path)
            text = "\n\n".join(doc.text for doc in documents)

            extracted_texts.append(
                f"\n===== FILE: {filename} =====\n{text}"
            )

        except Exception as e:
            print(f"❌ Failed to parse {filename}: {e}")

    return "\n\n".join(extracted_texts)


# =====================================================
# LLAMAINDEX PARSING SETUP
# =====================================================
from llama_index.core import (
    VectorStoreIndex,
    Document,
    PromptTemplate,
    Settings
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# -------------------------
# LLM (Gemini)
# -------------------------
Settings.llm = Gemini(
    model="models/gemini-2.5-flash",
    api_key="AIzaSyAx8P2QdFtxo08YZxlHc8mbsQJhiYwgMps"
)

# -------------------------
# Embeddings (FIX 🔥)
# -------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# SCHEMA
# =====================================================
DOCUMENT_SCHEMA = {
    "identity_verification": {
        "name": "",
        "address": ""
    },
    "credit_information": {
        "cibil_score": ""
    },
    "income_verification": {
        "applicant_income": ""
    }
}

# =====================================================
# PROMPT
# =====================================================
PARSING_PROMPT = PromptTemplate(
    """
You are a bank document parsing assistant.

Extract ONLY factual information explicitly present in the document.
DO NOT infer, assume, or guess.

Rules:
- If a field is not present, leave it as an empty string
- Return ONLY valid JSON
- Follow the schema EXACTLY

Schema:
{schema}

Document text:
{context_str}
"""
)

# =====================================================
# PARSE FUNCTION
# =====================================================
def parse_extracted_text(extracted_text):
    docs = [Document(text=extracted_text)]

    index = VectorStoreIndex.from_documents(docs)

    query_engine = index.as_query_engine(
        text_qa_template=PARSING_PROMPT.partial_format(
            schema=DOCUMENT_SCHEMA
        ),
        response_mode="compact"
    )

    response = query_engine.query(
        "Extract applicant information strictly as JSON."
    )

    return response.response


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    full_text = extract_all_documents_text(DOCUMENTS_DIR)

    print("\n================ EXTRACTED TEXT ================\n")
    print(full_text)

    print("\n================ PARSED STRUCTURED DATA ================\n")
    parsed_json = parse_extracted_text(full_text)
    print(parsed_json)
