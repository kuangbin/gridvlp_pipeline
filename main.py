from typing import Union, Annotated

from fastapi import FastAPI, Form

from modelscope.pipelines.multi_modal.gridvlp_pipeline import GridVlpClassificationPipeline

from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
  pipeline = GridVlpClassificationPipeline('rgtjf1/multi-modal_gridvlp_classification_chinese-base-ecom-cate-large')
  output = pipeline({'text': '女装快干弹力轻型短裤448575'})
  return output['text'][0]

class Data(BaseModel):
  text: str

@app.post("/type/")
def get_type(data: Data):
  pipeline = GridVlpClassificationPipeline('rgtjf1/multi-modal_gridvlp_classification_chinese-base-ecom-cate-large')
  output = pipeline({'text': data.text})
  return output['text'][0]
