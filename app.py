import os 
from fastapi import FastAPI , HTTPException,Form,Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn 
import joblib
import numpy as np



app = FastAPI()

# load the model from 
model = None
model_path  = os.path.join(os.path.dirname(__file__) , "ml model" , "wine_predictor.joblib")


try:
    model = joblib.load(model_path)
    print(f"model loaded successfully at : {model_path}")
    
except FileNotFoundError:
    print(f"model not found at :{model_path}")
    
except Exception as e :
    print(f"an error occured while loading model {e}")
    
    
    
# static files 
app.mount("/static" , StaticFiles(directory = "static"), name="static")


# set templates directory 
templates = Jinja2Templates(directory = "templates")

@app.get("/" , response_class= HTMLResponse)
def read_root(request :Request):
    return templates.TemplateResponse("index.html",{"request":request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, alcohol: float = Form(...), malic_acid: float = Form(...), ash: float = Form(...), alcalinity_of_ash: float = Form(...),magnesium: float = Form(...), total_phenols: float = Form(...), flavanoids: float = Form(...), nonflavanoid_phenols: float = Form(...), proanthocyanins: float = Form(...), color_intensity	: float = Form(...), hue: float = Form(...), od315_of_diluted_wines: float =Form(...), proline: float = Form(...)):
    if model is None:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Model not loaded"})

    data = np.array([alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols,proanthocyanins, color_intensity, hue, od315_of_diluted_wines, proline]).reshape(1, -1)

    try:
        prediction = model.predict(data)[0]
        class_mapping = {0: 'class1', 1: 'class2', 2: 'class3'}
        classname = class_mapping.get(prediction, "unknown")
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error making prediction: {e}"})
    
    return templates.TemplateResponse("index.html", {"request": request, "classname": classname})




class wineRequest(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium : float
    total_phenols : float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity : float
    hue : float
    od315_of_diluted_wines : float
    proline: float    
    
    
    
class wineResponse(BaseModel):
    classname: str

@app.post("/api/predict", response_model=wineResponse)
async def api_predict(wine: wineRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data = np.array([wine.alcohol, wine.malic_acid, wine.ash, wine.alcalinity_of_ash, wine.magnesium, wine.total_phenols, wine.flavanoids, wine.nonflavanoid_phenols,wine.proanthocyanins, wine.color_intensity, wine.hue, wine.od315_of_diluted_wines, wine.proline]).reshape(1, -1)

    try:
        prediction = model.predict(data)[0]
        class_mapping = {0: 'class1', 1: 'class2', 2: 'class3'}
        classname = class_mapping.get(prediction, "unknown")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")
    
    return wineResponse(classname=classname)




if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))  # Get the port from the environment variable or default to 8000
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")

    
    
