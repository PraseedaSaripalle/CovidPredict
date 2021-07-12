from django import forms

CHOICES= [
    ('Yes', 'Yes'),
    ('No', 'No')
    ]
diseases=[
    ('High BP-2','High BP-2'),
    ('Asthama-3','Asthama-3'),
    ('None','None')]

freq=[
    ("Sometimes",'Sometimes'),
    ('Often','Often'),
    ('Always','Always'),
    ('Rarely','Rarely')
]
times=[
    ("All of the time",'All of the time'),
    ('Most of the time','Most of the time'),
    ('Some of the time','Some of the time'),
    ('None of the time','None of the time')
]
gender=[
    ("Male","Male"),
    ("Female","Female")
]
age=[
    ("18-25","18-25"),
    ("26-32","26-32"),
    ("33-45","33-45"),
    (">45",">45")
]

class InputForm(forms.Form):
    eye_symptom = forms.CharField(label="Do You Have Eye Pain?", widget=forms.Select(choices=CHOICES))
    chest_symptom = forms.CharField(label="Do You Have Chest Pain?", widget=forms.Select(choices=CHOICES))
    soar_symptom=forms.CharField(label="Do You Have Soar Throat?", widget=forms.Select(choices=CHOICES))
    runnynose_symptom = forms.CharField(label="Do You Have Stuffy/runny nose?", widget=forms.Select(choices=CHOICES))
    weakness_symptom = forms.CharField(label="Do You Have Weakness/fatigue?", widget=forms.Select(choices=CHOICES))
    aches_symptom= forms.CharField(label="Do You Have Aches/Muscle pain?", widget=forms.Select(choices=CHOICES))
    headache_symptom = forms.CharField(label="Do You Have Headache?", widget=forms.Select(choices=CHOICES))
    cough_symptom=forms.CharField(label="Do You have cough?", widget=forms.Select(choices=CHOICES))
    breathing_symptom=forms.CharField(label="Do You face difficulty in breathing?", widget=forms.Select(choices=CHOICES))
    sleep_symptom=forms.CharField(label="Do You face change in sleep cycle?", widget=forms.Select(choices=CHOICES))
    meetppl=forms.CharField(label="Do you personally k0w anyone in your local community who is ill with a fever and either a cough or difficulty breathing?", widget=forms.Select(choices=CHOICES))
    corona_test=forms.CharField(label="Have you ever been tested for coronavirus (COVID-19)?", widget=forms.Select(choices=CHOICES))
    corona14=forms.CharField(label="Have you been tested for coronavirus (COVID-19) in the last 14 days?", widget=forms.Select(choices=CHOICES))
    disease = forms.CharField(label="Do you have any mentioned diseases?", widget=forms.Select(choices=diseases))
    gone_out = forms.CharField(label="In the last 24 hrs, have to gone to work outside the place where you live?", widget=forms.Select(choices=CHOICES))
    gone_out1=forms.CharField(label="In the last 24 hrs, have to gone to Supermarket or Pharmacy?", widget=forms.Select(choices=CHOICES))
    gone_out2 = forms.CharField(label="In the last 24 hrs, have to gone to Restaurant/CAFE/Shopping Mall?", widget=forms.Select(choices=CHOICES))
    spent_time=forms.CharField(label="In the last 24 hrs,have spent time with anyone else,who doesn't stay with you ?", widget=forms.Select(choices=CHOICES))
    more_meet = forms.CharField(label="In the last 24 hrs, have you attended a public meet?", widget=forms.Select(choices=CHOICES))
    crowd=forms.CharField(label="In the last 24 hrs, have been in a crowded place?", widget=forms.Select(choices=CHOICES))
    transport = forms.CharField(label="In the last 24 hrs, have used public transport?", widget=forms.Select(choices=CHOICES))
    home = forms.CharField(label="In the last 24 hrs, were you at home?", widget=forms.Select(choices=CHOICES))
    any = forms.CharField(label="In the last 24 hrs, have you done any of the following above mentioned things?", widget=forms.Select(choices=CHOICES))
    precautions = forms.CharField(label="In the last 24 hrs, were you using mask/hand sanitizer whenever you step out of your house?", widget=forms.Select(choices=freq))
    times=forms.CharField(label="How are you intentionally avoiding contact with other people?", widget=forms.Select(choices=times))
    gender = forms.CharField(label="Gender?", widget=forms.Select(choices=gender))
    smoke = forms.CharField(label="Do you smoke?", widget=forms.Select(choices=CHOICES))
    alc = forms.CharField(label="Do you comsume alcohol?", widget=forms.Select(choices=CHOICES))
    handwash = forms.CharField(label="Do you have access to handwash/sanitizer at work place?", widget=forms.Select(choices=CHOICES))
    handwash2 = forms.CharField(label="Do you have access to handwash/sanitizer at home?",widget=forms.Select(choices=CHOICES))
    age = forms.CharField(label="Age Band", widget=forms.Select(choices=age))





