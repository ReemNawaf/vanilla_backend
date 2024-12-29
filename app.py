import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from computer_vision.recommendation import recommend_outfit
# from computer_vision.segmentation import segment_image

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Mount images Files
app.mount("/images", StaticFiles(directory="images"), name="images")

apiUrl = 'http://0.0.0.0:8000'
user_img_path_recommend = 'images/upload/recommendation/user_image.jpg'
user_img_path_segment = 'images/upload/segmentation/user_image.jpg'


# ===| FastAPI Routes |===

# Home route
@app.get("/", response_class=HTMLResponse)
async def home():
    return JSONResponse(content= {"message": "Welcome to the Fashion Recommendation API!"}, status_code=200)
   

# ===| Recommendation route |===
@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    return recommend_outfit()

# ===| Segmentation route |===
# @app.post("/segment")
# async def segmentation():
#     return segment_image()

# ===| Upload Image Recommendation route |===
@app.post("/upload_image_recommend")
async def upload_image_recommend(file: UploadFile = File(...)):

    # Save uploaded file
    with open(user_img_path_recommend, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(content={'imageUrl': f'{apiUrl}/{user_img_path_recommend}'}, status_code=200)

# ===| Upload Image Segmentation route |===
@app.post("/upload_image_segment")
async def upload_image_segment(file: UploadFile = File(...)):
    
    # Save uploaded file
    with open(user_img_path_segment, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(content={'imageUrl': f'{apiUrl}/{user_img_path_segment}'}, status_code=200)

# ===| Start the FastAPI app |===
# uvicorn.run('app', host='0.0.0.0', port=8000, workers=2)
