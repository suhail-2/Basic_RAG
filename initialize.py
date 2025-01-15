import os
from dotenv import load_dotenv
import models

llm = models.ChatViaGroq(model_name= "mixtral-8x7b-32768")
embedding_model = "s"

response = llm.invoke("hello")
print(response.content)