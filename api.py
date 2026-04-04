from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import uuid
import gc

app = FastAPI(title="Nutricare Food API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = 'best_food_model_27classes.pt'
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Model failed to load: {e}")

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
    temp_file = f"temp_{uuid.uuid4()}_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    INFERENCE_SIZE = 640
    # Added iou=0.5 to prevent overlapping boxes natively
    results = model(temp_file, conf=0.273, imgsz=INFERENCE_SIZE, max_det=30, iou=0.5)
    
    meal_items = {}
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            w_norm = float(box.xywhn[0][2]) 
            h_norm = float(box.xywhn[0][3])
            area_ratio = w_norm * h_norm 
            
            # Portion scale logic based on bounding box size
            if area_ratio < 0.15:
                portion_scale = 0.5
            elif area_ratio > 0.45:
                portion_scale = 1.5
            else:
                portion_scale = 1.0

            macros = NUTRITION_DB.get(class_name, {'weight_g': 0, 'calories': 0, 'carbs': 0, 'protein': 0, 'fat': 0, 'sugar': 0, 'fiber': 0})
            
            # Calculate exactly what this specific box is worth
            item_weight = macros['weight_g'] * portion_scale
            item_cals = macros['calories'] * portion_scale
            item_protein = macros['protein'] * portion_scale
            item_fat = macros['fat'] * portion_scale
            item_carbs = macros['carbs'] * portion_scale
            item_sugar = macros['sugar'] * portion_scale
            item_fiber = macros['fiber'] * portion_scale

            # ACCUMULATION LOGIC: Add to existing totals if we already saw this food
            if class_name in meal_items:
                meal_items[class_name]['count'] += 1
                meal_items[class_name]['weight_g'] += item_weight
                meal_items[class_name]['calories'] += item_cals
                meal_items[class_name]['protein'] += item_protein
                meal_items[class_name]['fat'] += item_fat
                meal_items[class_name]['carbs'] += item_carbs
                meal_items[class_name]['sugar'] += item_sugar
                meal_items[class_name]['fiber'] += item_fiber
            else:
                meal_items[class_name] = {
                    "name": class_name,
                    "count": 1,
                    "weight_g": item_weight,
                    "calories": item_cals,
                    "protein": item_protein,
                    "fat": item_fat,
                    "carbs": item_carbs,
                    "sugar": item_sugar,
                    "fiber": item_fiber
                }
            
    os.remove(temp_file)
    
    # Process the dictionary into the final lists for FlutterFlow
    final_results = []
    total_calories = 0
    total_carbs = 0
    total_protein = 0
    total_fat = 0
    total_sugar = 0
    total_fiber = 0

    for key, item in meal_items.items():
        total_calories += int(round(item['calories'], 0))
        total_carbs += int(round(item['carbs'], 0))
        total_protein += int(round(item['protein'], 0))
        total_fat += int(round(item['fat'], 0))
        total_sugar += int(round(item['sugar'], 0))
        total_fiber += int(round(item['fiber'], 0))
        
        # Change the display name if there are multiple (e.g., "x2 Curry Puff")
        display_name = f"x{item['count']} {item['name']}" if item['count'] > 1 else item['name']
        
        final_results.append({
            "name": display_name,
            "amount": f"{int(round(item['weight_g'], 0))} g",
            "calories": int(round(item['calories'], 0)),
            "protein": int(round(item['protein'], 0)),
            "fat": int(round(item['fat'], 0)),
            "carbs": int(round(item['carbs'], 0)),
            "sugar": int(round(item['sugar'], 0)),
            "fiber": int(round(item['fiber'], 0))
        })
        
    # Clear RAM before responding
    del results
    gc.collect()
        
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
