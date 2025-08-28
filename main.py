from fastapi import FastAPI, UploadFile, File ,Request,Form
from fastapi.responses import HTMLResponse, Response,JSONResponse ,FileResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import base64
import json
from enum import Enum
from typing import List
import uvicorn



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # fallback for local
    uvicorn.run("main:app", host="0.0.0.0", port=port)


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/",response_class=HTMLResponse)
def home(request:Request):
  return templates.TemplateResponse("index.html",{"request":request})

"""class Regiontype(str,Enum):
   IGNORE : "ignore"
   SELECT : "select"
   """


@app.post("/compare")
async def compare_images(
   file1: UploadFile = File(...),
   file2: UploadFile = File(...),
   
   regions:str=Form("[]")
   ):
   try:
      img1_byte = await file1.read()
      img2_byte = await file2.read()
      img1_array=np.frombuffer(img1_byte,np.uint8)
      img2_array=np.frombuffer(img2_byte,np.uint8)
      img1_cv = cv2.imdecode(img1_array,cv2.IMREAD_COLOR)
      img2_cv = cv2.imdecode(img2_array,cv2.IMREAD_COLOR)
      if img1_cv.shape != img2_cv.shape:#type:ignore
         return JSONResponse(content={
            "status":"error",
            "message":"Images must be of same size"
         },status_code=400)
      img_h,img_w = img1_cv.shape[:2]#type:ignore
      gray1 = cv2.cvtColor(img1_cv,cv2.COLOR_BGR2GRAY)#type:ignore
      gray2 = cv2.cvtColor(img2_cv,cv2.COLOR_BGR2GRAY)#type:ignore
      diff = cv2.absdiff(gray1,gray2)
      
     
      mask = np.ones((img_h,img_w),dtype=np.uint8)*255
      region_data =json.loads(regions)
      if region_data:
         has_select =any(r["type"]=="select"for r in region_data)
         if has_select:
            mask[:]=0
         for region in region_data:
            x= int(region["x"])   
            y = int(region["y"])
            width =int(region["width"])
            height = int(region["height"])
            if region["type"]=="ignore":
               mask[y:y+height,x:x+width] =0
            elif region["type"]=="select" :
               mask[y:y+height,x:x+width]=255
         diff[mask==0]=0

      if np.count_nonzero(diff)==0:
         return JSONResponse(content={
            "status":"identical",
            "messege":"Both images are identical",
            "image":None
         },status_code=200)
      _,thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY)
      coords = np.column_stack(np.where(thresh>0))
      img2_pil = Image.fromarray(cv2.cvtColor(img2_cv, cv2.COLOR_BGR2RGB))#type:ignore
      draw = ImageDraw.Draw(img2_pil)
      for(y,x) in coords:
         draw.ellipse((x-1,y-1,x+1,y+1),fill=(255, 192, 203))

      out_bytes = io.BytesIO()
      img2_pil.save(out_bytes, format="PNG")
      out_bytes.seek(0)
      png_img = out_bytes.read()
      img_base64 = base64.b64encode(png_img).decode("utf-8")
      return JSONResponse(content={
         "status":"Different",
         "messege":"Both images are Different",
         "image":f"data:image/png;base64,{img_base64}"
      },status_code=200)
   except Exception as e:
      return JSONResponse(content={
         "status":"error",
         "messege": str(e)

      },status_code=400)



  
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")        

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_check():
    return Response(status_code=204)

        
    

   



 

   

