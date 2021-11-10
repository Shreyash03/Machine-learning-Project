import tkinter as tk
from tkinter import *
import PIL
from PIL import Image, ImageTk
import pandas as pd
import numpy as np



l1=['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 
'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 
'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 
'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 
'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 
'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 
'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 
'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 
'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 
'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 
'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 
'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 
'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 
'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 
'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 
'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 
'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 
'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']


l3=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','swelling_of_stomach',
'blurred_and_distorted_vision','phlegm',
'redness_of_eyes','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching',
'depression','irritability','muscle_pain','red_spots_over_body','belly_pain',
'abnormal_menstruation','watering_from_eyes','increased_appetite','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','stomach_bleeding',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
]

l5 =['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','swelling_of_stomach',
'blurred_and_distorted_vision','phlegm',
'redness_of_eyes','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching',
'depression','irritability','muscle_pain','red_spots_over_body','belly_pain',
'abnormal_menstruation','watering_from_eyes','increased_appetite',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','stomach_bleeding',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
]

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']
# print(len(l1), len(disease))

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

l4 = []
for x in range(0, len(l3)):
    l4.append(0)

l6 = []
for x in range(0, len(l5)):
    l6.append(0)



tr = pd.read_csv("Training.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X = tr[l1]
X_train = X[0::35]
Y = tr['prognosis']
np.ravel(Y)
Y_train = Y[0::35]
X1 = tr[l5]
X2 = tr[l3]

ts = pd.read_csv("Testing.csv")
ts.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

x = ts[l1]
x_test = X[::7]
y = ts['prognosis'] 
np.ravel(y)
y_test = Y[::7]
x1_test = ts[l5]
x2_test = ts[l3]
# y1_test =ts['prognosis']
# np.ravel(y1_test)




def knnClassifier():
    

    from sklearn.neighbors import KNeighborsClassifier
    model1 = KNeighborsClassifier(n_neighbors= 5)
    model1.fit(X_train, Y_train)
    y_pred = model1.predict(x)

    from sklearn.metrics import accuracy_score
    acc1 = accuracy_score(y, y_pred)
    print(f"Accuracy of the KNN model is: {acc1}")
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = model1.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break
    
    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def DecisionTree():
    

    from sklearn import tree

    model2 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    model2 = model2.fit(X1, np.ravel(Y))
    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=model2.predict(x1_test)
    acc2 = accuracy_score(y, y_pred)
    print(f"Accuracy of the Decision Tree algorithm is {acc2}")
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l5)):
        # print (k,)
        for z in psymptoms:
            if(z==l5[k]):
                l6[k]=1

    inputtest = [l6]
    predict = model2.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")
        

def NaiveBayes():
    
    from sklearn.naive_bayes import GaussianNB
    model3 = GaussianNB()
    model3 = model3.fit(X2, np.ravel(Y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=model3.predict(x2_test)
    acc3 = accuracy_score(y, y_pred)
    print(f"Accuracy of the Naive-Bayes algorithm is {acc3}")
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l3)):
        for z in psymptoms:
            if(z==l3[k]):
                l4[k]=1

    inputtest = [l4]
    predict = model3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")



window = Tk()
window.configure(background='grey')


Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()


image2 = Image.open(r'A:\DP\dp7.jpg')
image1 = ImageTk.PhotoImage(image2)
img = tk.Label(window, image = image1)
img.pack()
img.place(x=0, y=0)

image3 = Image.open(r'A:\DP\RIT.png')
image4 = image3.resize((90,92))
image5 = ImageTk.PhotoImage(image4)
img = tk.Label(window, image = image5)
img.pack()
img.place(x=200, y=20)

message = tk.Label(window,text="Disease-Prediction using Machine Learning", bg="#041212", fg="#7EF9FF", width=50,
                   height=2, font=('times new roman', 30, ' bold '))

message1 = tk.Label(window,text="By- G15", bg="#041212", fg="#ffffff", width=50,
                   height=2, font=('times new roman', 14, ' bold '))

message.place(x=300, y=20)
message1.place(x=500, y=120)


NameLb = tk.Label(window, text="Name of the patient", width=20, height=2, fg="#e54344", bg="#041212", font=('times', 15, ' bold '))
NameLb.place(x=250, y=200)

S1Lb = tk.Label(window, text="Symptom 1", width=20, height=2, fg="#e54344", bg="#041212", font=('times', 12, ' bold '))
S1Lb.place(x=850, y=200)

S2Lb = tk.Label(window, text="Symptom 2", width=20, height=2, fg="#e54344", bg="#041212", font=('times', 12, ' bold '))
S2Lb.place(x=850, y=250)

S3Lb = tk.Label(window, text="Symptom 3", width=20, height=2, fg="#e54344", bg="#041212", font=('times', 12, ' bold '))
S3Lb.place(x=850, y=300)

S4Lb = tk.Label(window, text="Symptom 4", width=20, height=2, fg="#e54344", bg="#041212", font=('times', 12, ' bold '))
S4Lb.place(x=850, y=350)

S5Lb = tk.Label(window, text="Symptom 5", width=20, height=2, fg="#e54344", bg="#041212", font=('times', 12, ' bold '))
S5Lb.place(x=850, y=400)


# entries
OPTIONS = sorted(l1)

NameEn = Entry(window, textvariable=Name)
NameEn.place(x=550, y=215)

S1En = OptionMenu(window, Symptom1,*OPTIONS)
S1En.place(x=1125, y=210)

S2En = OptionMenu(window, Symptom2,*OPTIONS)
S2En.place(x=1125, y=260)

S3En = OptionMenu(window, Symptom3,*OPTIONS)
S3En.place(x=1125, y=310)

S4En = OptionMenu(window, Symptom4,*OPTIONS)
S4En.place(x=1125, y=360)

S5En = OptionMenu(window, Symptom5,*OPTIONS)
S5En.place(x=1125, y=410)




prediction1 = tk.Button(window, text="KNN Model", command =knnClassifier , fg="#041212" ,bg="#ffffff"  ,width=20  ,height=2, activebackground = "#e54344" ,font=('times', 15, ' bold '))

prediction2 = tk.Button(window, text="Decision Tree", command =DecisionTree , fg="#041212" ,bg="#ffffff"  ,width=20  ,height=2, activebackground = "#e54344" ,font=('times', 15, ' bold '))

prediction3 = tk.Button(window, text="Naive Bayes", command =NaiveBayes , fg="#041212" ,bg="#ffffff"  ,width=20  ,height=2, activebackground = "#e54344" ,font=('times', 15, ' bold '))

prediction1.place(x=450, y=500)
prediction2.place(x=450, y=600)
prediction3.place(x=450, y=700)



t1 = Text(window, height=2, width=40,bg="#ffffff",fg="#041212")
t1.place(x=750, y=512.5)

t2 = Text(window, height=2, width=40,bg="#ffffff",fg="#041212")
t2.place(x=750, y=612.5)

t3 = Text(window, height=2, width=40,bg="#ffffff",fg="#041212")
t3.place(x=750, y=712.5)





window.mainloop()
