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
    return Response(ml.setPredictvalue(request.data["ema1"], request.data["ema2"], request.data["ema3"], request.data["ema4"]))
    # return Response(request.data)