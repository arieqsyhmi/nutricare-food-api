from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os

app = FastAPI(title="Nutricare Food API v5 (Anti-Duplicate)")

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
    pass

NUTRITION_DB = {
    'Ayam Goreng': {'weight_g': 154, 'calories': 436, 'carbs': 5.6, 'protein': 34.6, 'fat': 30.7}, 
    'Char Kway Teow': {'weight_g': 200, 'calories': 346, 'carbs': 48.0, 'protein': 9.0, 'fat': 13.0}, 
    'Cucur Udang': {'weight_g': 90, 'calories': 240, 'carbs': 31.4, 'protein': 5.2, 'fat': 10.4}, 
    'Curry Puff': {'weight_g': 48, 'calories': 153, 'carbs': 20.7, 'protein': 2.3, 'fat': 6.8}, 
    'Daging Rendang': {'weight_g': 50, 'calories': 127, 'carbs': 3.9, 'protein': 9.1, 'fat': 8.3}, 
    'Durian': {'weight_g': 54, 'calories': 83, 'carbs': 15.0, 'protein': 1.5, 'fat': 1.8}, 
    'Guava': {'weight_g': 150, 'calories': 69, 'carbs': 15.0, 'protein': 1.7, 'fat': 0.3}, 
    'Ikan Bakar': {'weight_g': 106, 'calories': 70, 'carbs': 0.4, 'protein': 11.9, 'fat': 2.2}, 
    'Kaya Toast': {'weight_g': 55, 'calories': 142, 'carbs': 30.0, 'protein': 3.4, 'fat': 1.0}, 
    'Kuih Lapis': {'weight_g': 100, 'calories': 152, 'carbs': 33.1, 'protein': 2.2, 'fat': 1.2}, 
    'Laksa Sarawak': {'weight_g': 300, 'calories': 355, 'carbs': 39.0, 'protein': 19.0, 'fat': 14.0}, 
    'Milo': {'weight_g': 250, 'calories': 277, 'carbs': 45.5, 'protein': 7.3, 'fat': 7.3}, 
    'Nasi Goreng Tambah Telur': {'weight_g': 261, 'calories': 470, 'carbs': 53.5, 'protein': 16.0, 'fat': 21.0}, 
    'Nasi Goreng': {'weight_g': 200, 'calories': 386, 'carbs': 53.0, 'protein': 10.0, 'fat': 15.0}, 
    'Nasi Kerabu': {'weight_g': 200, 'calories': 338, 'carbs': 51.0, 'protein': 8.0, 'fat': 11.0}, 
    'Nasi Lemak': {'weight_g': 200, 'calories': 338, 'carbs': 51.0, 'protein': 8.0, 'fat': 11.0}, 
    'Pisang Goreng': {'weight_g': 66, 'calories': 131, 'carbs': 24.7, 'protein': 1.3, 'fat': 2.9}, 
    'Roti Canai': {'weight_g': 84, 'calories': 266, 'carbs': 40.0, 'protein': 6.0, 'fat': 9.0}, 
    'Satay Ayam': {'weight_g': 15, 'calories': 34, 'carbs': 0.3, 'protein': 6.2, 'fat': 0.9}, 
    'Teh Tarik': {'weight_g': 250, 'calories': 220, 'carbs': 36.1, 'protein': 5.8, 'fat': 5.8}, 
    'Cucumber': {'weight_g': 20, 'calories': 3, 'carbs': 0.7, 'protein': 0.1, 'fat': 0.0}, 
    'Telur': {'weight_g': 61, 'calories': 94, 'carbs': 0.5, 'protein': 7.9, 'fat': 6.8}, 
    'Shrimp': {'weight_g': 24, 'calories': 12, 'carbs': 0.1, 'protein': 2.7, 'fat': 0.0}, 
    'Lime': {'weight_g': 10, 'calories': 5, 'carbs': 1.5, 'protein': 0.1, 'fat': 0.0},
    'Anchovies': {'weight_g': 12, 'calories': 30, 'carbs': 0.1, 'protein': 7.0, 'fat': 0.3}, 
    'Peanuts': {'weight_g': 12, 'calories': 65, 'carbs': 2.0, 'protein': 3.1, 'fat': 5.0}, 
    'Sambal': {'weight_g': 20, 'calories': 44, 'carbs': 5.0, 'protein': 1.0, 'fat': 2.0} 
}

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    results = model(temp_file, conf=0.28)
    
    # NEW: A dictionary to hold only the biggest version of each detected food
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

            macros = NUTRITION_DB.get(class_name, {'weight_g': 0, 'calories': 0, 'carbs': 0, 'protein': 0, 'fat': 0})
            
            # Save the best version to our dictionary
            unique_detections[class_name] = {
                "food_name": class_name,
                "detected_size": size_label,
                "bounding_box_area": round(area_ratio, 2),
                "estimated_weight_g": round(macros['weight_g'] * portion_scale, 1),
                "estimated_carbs": round(macros['carbs'] * portion_scale, 1),
                "estimated_calories": round(macros['calories'] * portion_scale, 1),
                "estimated_protein": round(macros['protein'] * portion_scale, 1),
                "estimated_fat": round(macros['fat'] * portion_scale, 1),
                "area": area_ratio # Hidden field just for the math logic
            }
            
    os.remove(temp_file)
    
    # Convert the dictionary back into a clean list for FlutterFlow
    final_results = [item for key, item in unique_detections.items()]
    
    # Remove the hidden math field before sending it to the app
    for item in final_results:
        del item['area']
        
    return {"success": True, "results": final_results}
