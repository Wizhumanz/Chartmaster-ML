from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from . import ml

# Create your views here.
"""
API Overview
"""
@api_view(['GET'])
def apiOverview(request):
    api_urls = {
        'List' : '/task-list/',
        'Detail View' : '/task-detail/<str:pk>/',
        'Create' : '/task-create/',
        'Update' : '/task-update/<str:pk>/',
        'Delete' : '/task-delete/<str:pk>/',
    }
    return Response(api_urls)

@api_view(['GET'])
def modelName(request):
    msg = "hello"
    return Response(msg)

@api_view(['POST'])
def sendInput(request):
    return Response(ml.setPredictvalue(request.data["volume1"], request.data["volume2"], request.data["volume3"], request.data["volatility"], request.data["volumeIndex"], request.data["time"], request.data["days"], request.data["months"]))
    # return Response(request.data)