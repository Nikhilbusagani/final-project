from django.shortcuts import render
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
import pandas as pd



# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})



def data_view(request):
    import pandas as pd
    from django.conf import settings
    import os
    path = os.path.join(settings.MEDIA_ROOT, 'ElectricCarData_Modified.csv')
    df = pd.read_csv(path)
    df = df.to_html
    return render(request, 'users/data.html', {'data': df})

def linear_regression(request):
    from .algorithms import AlgorithmUtility
    r2_score,mae,mse,rmse = AlgorithmUtility.calc_linear_regression()
    return render(request, 'users/linear.html',
                  {'r2_score': r2_score, "mae": mae, "mse": mse, "rmse": rmse})

def support_vector_classifier(request):
    from .algorithms import AlgorithmUtility
    r2_score,mae,mse = AlgorithmUtility.calc_support_vector_classifier()
    print(r2_score)
    return render(request, 'users/svm.html',
                  {'r2_score': r2_score, "mae": mae, "mse": mse})


def random_forest(request):
    from .algorithms import AlgorithmUtility
    r2_score,mae,mse = AlgorithmUtility.calc_random_forest()
    
    return render(request, 'users/rf.html',
                  {'r2_score': r2_score, "mae": mae, "mse": mse})

def decision_tree(request):
    from .algorithms import AlgorithmUtility
    r2_score,mae,mse = AlgorithmUtility.calc_decision_tree()
    print(r2_score)
    return render(request, 'users/dt.html',
                  {'r2_score': r2_score, "mae": mae, "mse": mse})

def ann(request):
    from .algorithms import AlgorithmUtility
    ann_loss, ann_mae = AlgorithmUtility.calc_perceptron_classifier()
    return render(request, 'users/ann.html',{'ann_loss': ann_loss, "ann_mae": ann_mae})


def user_Prediction(request):
    if request.method == "POST":
        AccelSec = int(request.POST.get("AccelSec"))
        TopSpeed_KmH =int(request.POST.get("TopSpeed_KmH"))
        Range_Km = int(request.POST.get("Range_Km"))
        Battery_Pack = int(request.POST.get("Battery_Pack"))
        Efficiency_WhKm = int(request.POST.get("Efficiency_WhKm"))
        FastCharge_KmH = int(request.POST.get("FastCharge_KmH"))
        RapidCharge = int(request.POST.get("RapidCharge"))
        PowerTrain = int(request.POST.get("PowerTrain"))
        PlugType = int(request.POST.get("PlugType"))
        BodyStyle = int(request.POST.get("BodyStyle"))
        Segment = int(request.POST.get("Segment"))
        Seats = int(request.POST.get("Seats"))
        
        # cohortType = int(request.POST.get("cohortType"))
        test = [AccelSec, TopSpeed_KmH, Range_Km, Battery_Pack,Efficiency_WhKm, FastCharge_KmH, RapidCharge, PowerTrain, PlugType, BodyStyle, Segment, Seats]
        print(test)
        from .algorithms import AlgorithmUtility
        rslt = AlgorithmUtility.test_user_date(test)
        
        return render(request, "users/Prediction_form.html", {"msg": rslt})
    else:
        return render(request, "users/Prediction_form.html", {})
