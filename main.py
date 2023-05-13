from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langchain.text_splitter import CharacterTextSplitter
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from langchain.llms import LlamaCpp
from langchain.llms import OpenAI

import argparse


BOOK_PATH = "three-pigs.txt"
MODEL_PATH = "WizardLM-7B-uncensored.ggml.q4_0.bin"
PROMPT_TEMPLATE = """You are book question and answer bot, you are provided with the following context to answer the question at the end. If you don't understand the question or don't know the answer, just say you don't know. Always give factual answer based on context provided.

Context:
{context}

Question: {question}
Answer:"""


def main(args):
    # Load the model
    if args.openai:
        # Load key from .env
        from dotenv import load_dotenv
        load_dotenv()
        llm = OpenAI(model_name="gpt-3.5-turbo")
    else:
        llm = LlamaCpp(
                model_path=args.model,
                n_ctx=1024
                )

    embeddings = HuggingFaceEmbeddings()

    # Prepate the data
    book_path = args.book
    with open(book_path, "r") as f:
        data = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_text(data)

    # Create the vector store
    docsearch = Chroma.from_texts(texts, embeddings)

    # Create the chain
    PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

    while query := input("Ask a question: "):
        docs = docsearch.similarity_search(query)

        output = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        print("Answer:")
        print(output["output_text"])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
            description="Run a question answering bot on a book"
            )
    argparser.add_argument("--openai", action="store_true",
                           help="Use OpenAI instead of Llama.cpp")
    argparser.add_argument("--model", type=str, default=MODEL_PATH,
                           help="Path to Llama.cpp the model")
    argparser.add_argument("--book", type=str, default=BOOK_PATH,
                           help="Path to the book .txt to use")
    args = argparser.parse_args()

    main(args)

