import pickle
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load the saved model
model = load_model('myapp/sentimental_analysis_model.h5')

# load the tokenizer used to preprocess the data
with open('myapp/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

def predict_sentiment(request):
    if request.method == 'POST':
        # get the input review from the POST request
        input_review = request.POST.get('input_review')
        # preprocess the input review
        input_seq = tokenizer.texts_to_sequences([input_review])
        input_padded = pad_sequences(input_seq, maxlen=100, padding='post', truncating='post')
        # make prediction using the trained model
        prediction = model.predict(input_padded)[0][0]
        # render the result in a template
        return render(request, 'result.html', {'prediction': prediction})
    else:
        return render(request, 'index.html')

