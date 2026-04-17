class Formatter:
    def __init__(self,app_income,coapp_income,place,cibil,emp_status,loan_amt,duration,dependents,marital_status,gender):
        self.app_income = app_income
        self.coapp_income = coapp_income
        self.place = place.strip().lower()
        self.cibil = self.cibil_quality()      
        self.emp_status = emp_status.strip().lower()
        self.loan_amt = loan_amt
        self.duration = duration
        self.emi_repayment_capacity = self.emi_repayment_capacity()
        self.dependents = self.dependents_quality(dependents)
        self.marital_status=marital_status.strip().lower()
        self.gender=gender.strip().lower()

    def emi_repayment_capacity(self):
        effective_income = (self.app_income + self.coapp_income) * 0.5
        emi = self.loan_amt / self.duration
        ratio = emi / effective_income

        if ratio <= 1.0:
            return "High"
        elif ratio <= 1.5:
            return "Moderate"
        else:
            return "Low"
        
    def cibil_quality(self):
        if self.cibil >= 750:
            return "Good"
        elif self.cibil >= 600:
            return "Average"
        else:
            return "Poor"
    def dependents_quality(self,num_dependents):
        if num_dependents <=1:
            return "Low"
        elif num_dependents == 2:
            return "Moderate"
        else:
            return "High"

    def get(self):
        return {
            "place": self.place,
            "cibil": self.cibil,
            "emp_status": self.emp_status,
            "emi_repayment_capacity": self.emi_repayment_capacity,
            "dependents": self.dependents,
            "marital_status":self.marital_status,
            "gender":self.gender
        }