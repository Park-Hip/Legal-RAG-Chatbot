from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import dotenv
from embeddings.embedder import get_vectorstore
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelName)s - %(message)s'
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

db = get_vectorstore()

rag_prompt_template = '''
You are a legal assistant trained to provide precise, formal legal responses.

When a user asks a question, you will be given:
- A retrieved legal excerpt (the "context")
- Metadata: article number, clause or point, chapter name, and the title of the legal document

Your task is to draft a **formal legal-style answer** that:
- Clearly answers the user’s question
- Refers specifically to the law by citing the **article number**, **clause/point**, **chapter**,
 and **title of the legal document**
- Follows a formal tone suitable for legal interpretation
- Does **not** speculate or fabricate any legal rules — rely solely on the given text and metadata
- If the context is insufficient, state that the answer cannot be determined from the available legal text.

---

**User Question:**  
{question}

**Retrieved Legal Context:**  
{context}

**Metadata:**  
- Article: {article_number}  
- Clause/Point: {point_number}  
- Chapter: {chapter_name}  
- Title: {document_title}

---

**Formal Legal Answer Format:**  
Begin your answer using a phrase such as:

> "Pursuant to Article {article_number}, Point {point_number}, Chapter {chapter_name}, of the {document_title}, ..."

Then provide a clear, formal interpretation of the legal text based strictly on the context.


'''

prompt = ChatPromptTemplate.from_messages([
    ('system', "You are a legal expert assisting individuals with real-life legal questions. "
               "Only answer based on the provided context. If unsure, say: The answer is not"
               "available in the provided context."),
    ('user', "Context:\n{context}\n\nQuestion:\n{input}\n\nAnswer:")
])

def get_rag_chain():

    llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=1000
)

    retriever = db.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k': 3}
    )

    combine_doc_chain = create_stuff_documents_chain(prompt=prompt, llm=llm)
    return create_retrieval_chain(retriever, combine_doc_chain)


rag_chain = get_rag_chain()
query = 'The offender is 70 years of age or older; Will his punishment mitigated ?'
response = rag_chain.invoke({'input': query})
logger.info(response['answer'])