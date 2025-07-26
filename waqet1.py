from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

app = FastAPI(title="WAQET Backend Prototype", version="1.0")


# نماذج البيانات

class SignupRequest(BaseModel):
    name: str
    birthdate: str
    national_id: str
    organization: str
    job_title: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class UserProfile(BaseModel):
    id: str
    name: str
    birthdate: str
    national_id: str
    organization: str
    job_title: str
    email: str

class FlightStatus(BaseModel):
    flight_number: str
    origin: str
    gate: str
    gpu_activated: bool

class ActivateGPURequest(BaseModel):
    flight_number: str
    gate: str

class PredictETARequest(BaseModel):
    distance_km: float
    speed_kmh: float

class EmissionRequest(BaseModel):
    apu_hours: float
    gpu_hours: float


# بيانات مؤقتة

users_db = []
sessions = {}
airports_data = {
    "RUH": ["Gate A1", "Gate A2", "Gate A3"],
    "JED": ["Gate B1", "Gate B2"],
    "DMM": ["Gate C1"]
}
flights_data = [
    {"flight_number": "XY101", "origin": "JED", "gate": "Gate A1", "gpu_activated": False},
    {"flight_number": "XY202", "origin": "DMM", "gate": "Gate A2", "gpu_activated": True},
]

notifications = [
    {"message": "فعل النظام للرحلة XY101 في البوابة Gate A1"},
    {"message": "فعل النظام للرحلة XY202 في البوابة Gate A2"}
]

# دوال مساعدة
def get_current_user(token: str):
    if token not in sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return sessions[token]

# Auth
@app.post("/auth/signup")
def signup(data: SignupRequest):
    for user in users_db:
        if user["email"] == data.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    new_user = data.dict()
    new_user["id"] = str(uuid.uuid4())
    users_db.append(new_user)
    return {"message": "User created successfully", "user_id": new_user["id"]}

@app.post("/auth/login")
def login(data: LoginRequest):
    for user in users_db:
        if user["email"] == data.email and user["password"] == data.password:
            token = str(uuid.uuid4())
            sessions[token] = user
            return {"token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/auth/profile", response_model=UserProfile)
def profile(token: str):
    user = get_current_user(token)
    return user

# Airports & Gates
@app.get("/airports/list")
def list_airports():
    return {"airports": list(airports_data.keys())}

@app.get("/airports/{airport_id}/gates")
def gates(airport_id: str):
    if airport_id not in airports_data:
        raise HTTPException(status_code=404, detail="Airport not found")
    return {"airport": airport_id, "gates": airports_data[airport_id]}

# Flights
@app.get("/flights/timeline", response_model=List[FlightStatus])
def flight_timeline():
    return flights_data

@app.get("/flights/{gate_id}/status")
def gate_status(gate_id: str):
    gate_flights = [f for f in flights_data if f["gate"] == gate_id]
    return {"gate": gate_id, "flights": gate_flights}

@app.post("/flights/activate_gpu")
def activate_gpu(req: ActivateGPURequest):
    for flight in flights_data:
        if flight["flight_number"] == req.flight_number and flight["gate"] == req.gate:
            flight["gpu_activated"] = True
            return {"message": f"GPU Activated for {req.flight_number} at {req.gate}"}
    raise HTTPException(status_code=404, detail="Flight not found")


# Notifications
@app.get("/notifications/technician")
def technician_notifications():
    return {"notifications": notifications}

# AI Mock Models
@app.post("/ai/predict_eta")
def predict_eta(req: PredictETARequest):
    if req.speed_kmh == 0:
        raise HTTPException(status_code=400, detail="Speed cannot be zero")
    eta = req.distance_km / req.speed_kmh
    return {"eta_hours": round(eta, 2)}

@app.post("/ai/emission_stats")
def emission_stats(req: EmissionRequest):
    apu_emission = req.apu_hours * 200 * 3.16
    gpu_emission = req.gpu_hours * 50 * 3.16
    saved = apu_emission - gpu_emission
    return {"apu_emission": apu_emission, "gpu_emission": gpu_emission, "saved_emission": saved}

# Dashboard
@app.get("/dashboard/daily_reports")
def daily_reports():
    return {
        "total_flights": len(flights_data),
        "notifications_sent": len(notifications),
        "gpu_activated": sum(1 for f in flights_data if f["gpu_activated"])
    }

@app.get("/dashboard/sustainability")
def sustainability():
    saved_co2 = 500  # mock value
    saved_fuel = 300  # mock value
    return {"co2_saved": saved_co2, "fuel_saved": saved_fuel}

# ===============================
# Admin Panel
# ===============================
@app.get("/admin/panel")
def admin_panel(token: str):
    user = get_current_user(token)
    if user["job_title"].lower() != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"users": users_db, "flights": flights_data}

# ===============================
# Static Pages
# ===============================
@app.get("/static/about")
def about():
    return {"about": "WAQET - Smart Airport Sustainability System"}

@app.get("/static/contact")
def contact():
    return {"contact": "Contact us at info@waqet.com"}

@app.get("/static/privacy")
def privacy():
    return {"privacy": "We respect user privacy and data security."}
if __name__ == "__main__":
    print("App file loaded correctly")
    
#  predict_eta.py
# نموذج لتنبؤ وقت وصول الطائرة (ETA) بناءً على المسافة والسرعة
# الهدف: تجهيز وحدات الطاقة الأرضية في الوقت المناسب لتقليل التشغيل غير الضروري
# التشغيل كبروتوتايب: ندخل المسافة والسرعة يدويًا، مستقبلًا ممكن الربط مع بيانات رحلات حقيقية
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

MODEL_FILE = "eta_model.pkl"

def train_eta_model():
    X = np.random.rand(1000, 2) * [1000, 900]
    y = X[:, 0] / X[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("ETA model MSE:", mean_squared_error(y_test, y_pred))
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

def predict_eta(distance_km, speed_kmh):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return float(model.predict(np.array([[distance_km, speed_kmh]]))[0])


#  estimate_fuel.py
# نموذج بسيط لتقدير استهلاك الوقود بناءً على مدة تشغيل وحدة الطاقة المساعدة (APU)
# الهدف: قياس كمية الوقود الممكن توفيرها عند تقليل تشغيل APU
# التشغيل كبروتوتايب: ندخل وقت التشغيل بالساعات ويُرجع كمية الوقود

def estimate_fuel(aputime_hours):
    return aputime_hours * 200  # استهلاك تقريبي 200 كجم/ساعة


#  estimate_emissions.py
# نموذج بسيط لحساب انبعاثات CO₂ الناتجة عن استهلاك الوقود
# الهدف: قياس الأثر البيئي لاستهلاك الوقود
# التشغيل كبروتوتايب: ندخل كمية الوقود ويُرجع الانبعاثات بالكيلوجرام

def estimate_emissions(fuel_kg):
    return fuel_kg * 3.16  # كل 1 كجم وقود ينتج 3.16 كجم CO₂


#  anomaly_detection.py
# نموذج اختياري لكشف التشغيل غير الطبيعي (Anomalies)
# الهدف: التعرف على القيم الشاذة في بيانات التشغيل
# التشغيل كبروتوتايب: يدخل بيانات تشغيل ويرجع 1 (طبيعي) أو -1 (شاذ)

import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    model = IsolationForest(contamination=0.05)
    model.fit(data)
    return model.predict(data)


# main.py
# ملف رئيسي لتجميع وتشغيل كل النماذج
# الهدف: تجربة كاملة من إدخال بيانات الرحلة إلى حساب الوقت والوقود والانبعاثات
# التشغيل كبروتوتايب: ندرب النموذج مرة واحدة ثم نستدعيه لإرجاع النتائج

def full_pipeline(distance_km, speed_kmh):
    eta = predict_eta(distance_km, speed_kmh)
    fuel = estimate_fuel(eta)
    emissions = estimate_emissions(fuel)
    return {"eta_hours": eta, "fuel_kg": fuel, "emissions_kg": emissions}

if __name__ == "__main__":
    train_eta_model()
    results = full_pipeline(500, 700)
    print(results)