#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain-community==0.2.4 langchain==0.2.3 faiss-cpu==1.8.0 unstructured==0.14.5 unstructured[pdf]==0.14.5 transformers==4.41.2 sentence-transformers==3.0.1')


# **Importing the Dependencies**

# In[19]:


import os

from langchain_community.llms import Ollama
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA


# In[20]:


# loading the LLM
llm = Ollama(
    model="llama3:instruct",
    temperature=0
)


# In[21]:


# loading the document
loader = UnstructuredFileLoader("Fogler CRE.pdf") 
documents = loader.load()


# In[22]:


# create document chunks
text_splitter = CharacterTextSplitter(separator="/n",
                                      chunk_size=1000,
                                      chunk_overlap=200)


# In[23]:


text_chunks = text_splitter.split_documents(documents)


# In[16]:


# loading the vector embedding model
embeddings = HuggingFaceEmbeddings()


# In[17]:


knowledge_base = FAISS.from_documents(text_chunks, embeddings)


# In[18]:


# retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=knowledge_base.as_retriever())


# In[17]:


question = "What is this document about?"
response = qa_chain.invoke({"query": question})
print(response["result"])


# In[ ]:





# In[ ]:





# In[ ]:




