from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import uuid  # Added to prevent race conditions during simultaneous user uploads

app = FastAPI(title="Nutricare Food API v6 (Fully Structured for FlutterFlow)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # This is the VIP pass that lets FlutterFlow connect!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = 'best_food_model_27classes.pt'
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Model failed to load: {e}")

# Complete Database with calculated Sugar and Fiber metrics for Diabetic Management
NUTRITION_DB = {
    'Ayam Goreng': {'weight_g': 154, 'calories': 436, 'carbs': 5.6, 'protein': 34.6, 'fat': 30.7, 'sugar': 0.1, 'fiber': 0.0}, 
    'Char Kway Teow': {'weight_g': 200, 'calories': 346, 'carbs': 48.0, 'protein': 9.0, 'fat': 13.0, 'sugar': 3.5, 'fiber': 2.0}, 
    'Cucur Udang': {'weight_g': 90, 'calories': 240, 'carbs': 31.4, 'protein': 5.2, 'fat': 10.4, 'sugar': 1.5, 'fiber': 1.2}, 
    'Curry Puff': {'weight_g': 48, 'calories': 153, 'carbs': 20.7, 'protein': 2.3, 'fat': 6.8, 'sugar': 1.0, 'fiber': 1.5}, 
    'Daging Rendang': {'weight_g': 50, 'calories': 127, 'carbs': 3.9, 'protein': 9.1, 'fat': 8.3, 'sugar': 2.5, 'fiber': 1.0}, 
    'Durian': {'weight_g': 54, 'calories': 83, 'carbs': 15.0, 'protein': 1.5, 'fat': 1.8, 'sugar': 14.0, 'fiber': 2.0}, 
    'Guava': {'weight_g': 150, 'calories': 69, 'carbs': 15.0, 'protein': 1.7, 'fat': 0.3, 'sugar': 13.0, 'fiber': 8.0}, 
    'Ikan Bakar': {'weight_g': 106, 'calories': 70, 'carbs': 0.4, 'protein': 11.9, 'fat': 2.2, 'sugar': 1.0, 'fiber': 0.0}, 
    'Kaya Toast': {'weight_g': 55, 'calories': 142, 'carbs': 30.0, 'protein': 3.4, 'fat': 1.0, 'sugar': 15.0, 'fiber': 1.5}, 
    'Kuih Lapis': {'weight_g': 100, 'calories': 152, 'carbs': 33.1, 'protein': 2.2, 'fat': 1.2, 'sugar': 18.0, 'fiber': 0.5}, 
    'Laksa Sarawak': {'weight_g': 300, 'calories': 355, 'carbs': 39.0, 'protein': 19.0, 'fat': 14.0, 'sugar': 4.0, 'fiber': 3.0}, 
    'Milo': {'weight_g': 250, 'calories': 277, 'carbs': 45.5, 'protein': 7.3, 'fat': 7.3, 'sugar': 20.0, 'fiber': 1.5}, 
    'Nasi Goreng Tambah Telur': {'weight_g': 261, 'calories': 470, 'carbs': 53.5, 'protein': 16.0, 'fat': 21.0, 'sugar': 3.0, 'fiber': 2.5}, 
    'Nasi Goreng': {'weight_g': 200, 'calories': 386, 'carbs': 53.0, 'protein': 10.0, 'fat': 15.0, 'sugar': 2.5, 'fiber': 2.0}, 
    'Nasi Kerabu': {'weight_g': 200, 'calories': 338, 'carbs': 51.0, 'protein': 8.0, 'fat': 11.0, 'sugar': 2.0, 'fiber': 4.0}, 
    'Nasi Lemak': {'weight_g': 200, 'calories': 338, 'carbs': 51.0, 'protein': 8.0, 'fat': 11.0, 'sugar': 2.0, 'fiber': 3.0}, 
    'Pisang Goreng': {'weight_g': 66, 'calories': 131, 'carbs': 24.7, 'protein': 1.3, 'fat': 2.9, 'sugar': 8.0, 'fiber': 1.5}, 
    'Roti Canai': {'weight_g': 84, 'calories': 266, 'carbs': 40.0, 'protein': 6.0, 'fat': 9.0, 'sugar': 1.5, 'fiber': 2.0}, 
    'Satay Ayam': {'weight_g': 15, 'calories': 34, 'carbs': 0.3, 'protein': 6.2, 'fat': 0.9, 'sugar': 2.0, 'fiber': 0.1}, 
    'Teh Tarik': {'weight_g': 250, 'calories': 220, 'carbs': 36.1, 'protein': 5.8, 'fat': 5.8, 'sugar': 22.0, 'fiber': 0.0}, 
    'Cucumber': {'weight_g': 20, 'calories': 3, 'carbs': 0.7, 'protein': 0.1, 'fat': 0.0, 'sugar': 0.3, 'fiber': 0.1}, 
    'Telur': {'weight_g': 61, 'calories': 94, 'carbs': 0.5, 'protein': 7.9, 'fat': 6.8, 'sugar': 0.2, 'fiber': 0.0}, 
    'Shrimp': {'weight_g': 24, 'calories': 12, 'carbs': 0.1, 'protein': 2.7, 'fat': 0.0, 'sugar': 0.0, 'fiber': 0.0}, 
    'Lime': {'weight_g': 10, 'calories': 5, 'carbs': 1.5, 'protein': 0.1, 'fat': 0.0, 'sugar': 0.2, 'fiber': 0.3},
    'Anchovies': {'weight_g': 12, 'calories': 30, 'carbs': 0.1, 'protein': 7.0, 'fat': 0.3, 'sugar': 0.0, 'fiber': 0.0}, 
    'Peanuts': {'weight_g': 12, 'calories': 65, 'carbs': 2.0, 'protein': 3.1, 'fat': 5.0, 'sugar': 0.5, 'fiber': 1.0}, 
    'Sambal': {'weight_g': 20, 'calories': 44, 'carbs': 5.0, 'protein': 1.0, 'fat': 2.0, 'sugar': 4.0, 'fiber': 0.8} 
}

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    # Added UUID to ensure safe multi-user processing
    temp_file = f"temp_{uuid.uuid4()}_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # =====================================================================
    # RENDER STRESS TEST: Increased Resolution
    # Default is 640. Trying 1024 for better small-object detection.
    # =====================================================================
    INFERENCE_SIZE = 1024
    results = model(temp_file, conf=0.28, imgsz=INFERENCE_SIZE)
    
    # A dictionary to hold only the biggest version of each detected food
    unique_detections = {}
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            w_norm = float(box.xywhn[0][2]) 
            h_norm = float(box.xywhn[0][3])
            area_ratio = w_norm * h_norm 
            
            # If we already saw this food, only overwrite it if this new box is bigger
            if class_name in unique_detections:
                if area_ratio <= unique_detections[class_name]['area']:
                    continue # Skip this smaller duplicate box
            
            if area_ratio < 0.15:
                portion_scale = 0.5
                size_label = "Small Portion"
            elif area_ratio > 0.45:
                portion_scale = 1.5
                size_label = "Large Portion"
            else:
                portion_scale = 1.0
                size_label = "Standard Portion"

            # Fallback dictionary now includes sugar and fiber
            macros = NUTRITION_DB.get(class_name, {'weight_g': 0, 'calories': 0, 'carbs': 0, 'protein': 0, 'fat': 0, 'sugar': 0, 'fiber': 0})
            
            # PERFECTLY MATCHED TO FLUTTERFLOW STRUCT (Integers for macros, String for amount)
            unique_detections[class_name] = {
                "name": class_name,
                "amount": f"{int(round(macros['weight_g'] * portion_scale, 0))} g",
                "calories": int(round(macros['calories'] * portion_scale, 0)),
                "protein": int(round(macros['protein'] * portion_scale, 0)),
                "fat": int(round(macros['fat'] * portion_scale, 0)),
                "carbs": int(round(macros['carbs'] * portion_scale, 0)),
                "sugar": int(round(macros['sugar'] * portion_scale, 0)), 
                "fiber": int(round(macros['fiber'] * portion_scale, 0)), 
                "area": area_ratio # Hidden field just for the math logic
            }
            
    os.remove(temp_file)
    
    # Convert the dictionary back into a clean list for FlutterFlow
    final_results = [item for key, item in unique_detections.items()]
    
    # Calculate the Totals (using the new integer keys)
    total_calories = sum(item['calories'] for item in final_results)
    total_carbs = sum(item['carbs'] for item in final_results)
    total_protein = sum(item['protein'] for item in final_results)
    total_fat = sum(item['fat'] for item in final_results)
    total_sugar = sum(item['sugar'] for item in final_results)
    total_fiber = sum(item['fiber'] for item in final_results)

    # Remove the hidden math field before sending it to the app
    for item in final_results:
        del item['area']
        
    # Send back BOTH the Totals and the Individual Results
    return {
        "success": True, 
        "totals": {
            "calories": total_calories,
            "carbs": total_carbs,
            "protein": total_protein,
            "fat": total_fat,
            "sugar": total_sugar,
            "fiber": total_fiber
        },
        "results": final_results
    }
