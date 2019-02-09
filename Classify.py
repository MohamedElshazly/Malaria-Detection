from starlette.applications import Starlette
from starlette.responses import JSONResponse,PlainTextResponse, HTMLResponse, RedirectResponse, TemplateResponse
from starlette.requests import Request
from fastai.vision import (
    ImageDataBunch,
    create_cnn,
    open_image,
    get_transforms,
    models,
    imagenet_stats,
)
from fastai.metrics import accuracy

from jinja2 import Environment, FileSystemLoader
import torch 
from io import BytesIO 
from pathlib import Path
import sys 
import uvicorn 
import aiohttp 
import asyncio 

env = Environment(loader= FileSystemLoader('templates'))

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read() 

app = Starlette() 

imgs_path = Path("tmp") 
classes = ["/{}_1.jpg".format(x) for x in ['Infected', 'Not Infected']]
pred_data = ImageDataBunch.from_name_re(imgs_path, 
                                        classes,
                                        r"/([^/]+)_\d+.jpg$",
                                        ds_tfms = get_transforms(),
                                        size = 224,
                                        num_workers = 0).normalize(imagenet_stats)

# defaults.device = torch.device('cpu')
pred_learn = create_cnn(pred_data, models.resnet50, metrics = accuracy).load('model-2')

@app.route("/upload", methods =["POST"]) 
async def upload(request):
    data = await request.form() 
    bytes = await (data["file"].read()) 
    return predict_class(bytes) 

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"]) 
    return predict_class(bytes) 

def predict_class(bytes):
    img = open_image(BytesIO(bytes))
    pred_class, _,_ = pred_learn.predict(img) 
    return PlainTextResponse("Prediction : {}".format(pred_class))

@app.route("/")
def form(request):
    template = env.get_template('index.html')
    context = {'request' : request} 
    response = TemplateResponse(template, context)
    return response

@app.route("/form")
def redirect_to_home(request):
    return RedirectResponse("/")

if __name__ == '__main__':
    # if "serve" in sys.argv:
    uvicorn.run(app, host = '0.0.0.0', port = 8008)


    