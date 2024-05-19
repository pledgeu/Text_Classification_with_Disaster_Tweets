## NLP with Disaster Tweets ##

The task is text classification.

Predict which Tweets are about real disasters and which ones are not.

---

In this notebook,I will introduce:
+ Necessary libraries

+ Load the dataset

+ Data Exploration

+ Preprocess the data

+ Train model

+ Generate the submission file

## Necessary libraries ##


```python
%matplotlib inline
import numpy as np
import pandas as pd
import torch
from torch import nn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import sklearn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt 
import seaborn as sns 
```

## Load the dataset ##

Train and test dataset contain:

+ id

+ keyword

+ location

+ text

+ target

This task is text classification. The target is 1, which means it is a real disaster, otherwise it is 0.


```python
import pandas as pd
train_data=pd.read_csv("D:/kaggle/input/nlp-getting-started/train.csv")
test_data=pd.read_csv("D:/kaggle/input/nlp-getting-started/test.csv")
```

## Data Exploration ##


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Heard about #earthquake is different cities, s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>there is a forest fire at spot pond, geese are...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check miss value
train_data.isnull().sum()
```




    id             0
    keyword       61
    location    2533
    text           0
    target         0
    dtype: int64



The dataset is fine.The task is text classification. Locations and keywords have little impact on the results. As long as the text and targets are not missing, it will be fine.


```python
train_data["target"].value_counts()
```




    0    4342
    1    3271
    Name: target, dtype: int64




```python
train_data["length"]=train_data["text"].apply(len)
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
      <td>69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
      <td>133</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
      <td>65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data['length'].describe()
```




    count    7613.000000
    mean      101.037436
    std        33.781325
    min         7.000000
    25%        78.000000
    50%       107.000000
    75%       133.000000
    max       157.000000
    Name: length, dtype: float64




```python
import matplotlib.pyplot as plt 

def plot_tweet_length_histogram(data):
    plt.figure(figsize=(9, 9))
    plt.hist(data["length"], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Length of Tweets")
    plt.xlabel("Length")
    plt.ylabel("Density")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


plot_tweet_length_histogram(train_data)
```


    
![png](output_14_0.png)
    


## Preprocess the data ##


```python
def preprocess_text(text):
    
    # Lowercasing
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Removing special characters and punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens]
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Removing URLs, replace by 'URL'
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    tokens = [re.sub(url_pattern, 'URL', token) for token in tokens]
    
    # Removing HTML beacons, assuming 'html' as a keyword
    tokens = [token for token in tokens if 'html' not in token]
    
    # Removing numbers, replace by 'number'
    tokens = [re.sub(r'\d+', 'number', token) for token in tokens]
    
    # Removing emojis, replace by 'emoji'
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    tokens = [re.sub(emoji_pattern, 'emoji', token) for token in tokens]
    
 
    return tokens

```


```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
train_data['text'] = train_data['text'].apply(preprocess_text)
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[deeds, reason, , earthquake, may, allah, forg...</td>
      <td>1</td>
      <td>69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[forest, fire, near, la, ronge, sask, , canada]</td>
      <td>1</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[residents, asked, shelter, place, , notified,...</td>
      <td>1</td>
      <td>133</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[number, people, receive, , wildfires, evacuat...</td>
      <td>1</td>
      <td>65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[got, sent, photo, ruby, , alaska, smoke, , wi...</td>
      <td>1</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data["length"]=train_data["text"].apply(len)
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[deeds, reason, , earthquake, may, allah, forg...</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[forest, fire, near, la, ronge, sask, , canada]</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[residents, asked, shelter, place, , notified,...</td>
      <td>1</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[number, people, receive, , wildfires, evacuat...</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[got, sent, photo, ruby, , alaska, smoke, , wi...</td>
      <td>1</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data['length'].describe()
```




    count    7613.000000
    mean       13.978327
    std         5.459585
    min         1.000000
    25%        10.000000
    50%        14.000000
    75%        18.000000
    max        67.000000
    Name: length, dtype: float64




```python
import matplotlib.pyplot as plt 

def plot_tweet_length_histogram(data):
    plt.figure(figsize=(9, 9))
    plt.hist(data["length"], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Length of Tweets")
    plt.xlabel("Length")
    plt.ylabel("Density")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Assuming train_data is your DataFrame containing tweet lengths
plot_tweet_length_histogram(train_data)
```


    
![png](output_20_0.png)
    


In order to improve computational efficiency, we will process small batches of text sequences through truncation and padding. Assuming that each sequence in the same mini-batch should have the same length num_steps, then if the number of tokens of the text sequence is less than num_steps, we will continue to add specific "<pad>" tokens at the end until it The length reaches num_steps; otherwise, when we truncate the text sequence, only take the first num_steps tokens and discard the remaining tokens. This way, each text sequence will be of the same length, allowing it to be loaded in mini-batches of the same shape.

According to this figure, in order to facilitate calculation, it is reasonable to set the maximum value of padding to 25.


```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# Apply preprocessing function to 'text' column in train_data and test_data
train_data['text'] = train_data['text'].apply(preprocess_text)
test_data['text'] = test_data['text'].apply(preprocess_text)

from collections import Counter
def build_vocab(texts):
    # Join all tokens into a single list and count occurrences
    token_counts = Counter([token for sublist in texts for token in sublist])
    # Create a word to index mapping (starting from 1, 0 is used for padding)
    vocab = {word: i+1 for i, (word, _) in enumerate(token_counts.items())}
    return vocab

def encode(tokens, vocab):
    return [vocab.get(token, 0) for token in tokens]  # 0 indexed for unknown words

def pad_text(encoded_texts, max_length):
    padded_seqs = [torch.tensor(text[:max_length] + [0]*(max_length - len(text))) for text in encoded_texts]
    return pad_sequence(padded_seqs, batch_first=True)

# Modify the code to incorporate the maximum length information
max_length = 25  # Set your desired maximum length here

# Create vocabularies
vocab = build_vocab(list(train_data['text']) + list(test_data['text']))

# Encode tokens using vocab
train_data['text_encoded'] = train_data['text'].apply(lambda x: encode(x, vocab))
test_data['text_encoded'] = test_data['text'].apply(lambda x: encode(x, vocab))

# Pad sequences to a max length
train_padded = pad_text(train_data['text_encoded'], max_length)
test_padded = pad_text(test_data['text_encoded'], max_length)
train_labels = torch.tensor(train_data['target'].values)

train_padded
```




    tensor([[   1,    2,    3,  ...,    0,    0,    0],
            [   9,   10,   11,  ...,    0,    0,    0],
            [  16,   17,   18,  ...,    0,    0,    0],
            ...,
            [1409,    3,   25,  ...,    0,    0,    0],
            [ 184, 4590, 6822,  ...,    0,    0,    0],
            [1471,    3, 5003,  ...,    0,    0,    0]])




```python
train_labels
```




    tensor([1, 1, 1,  ..., 1, 1, 1])




```python
from collections import Counter
token_counts=Counter([token for sublist in list(train_data['text']) for token in sublist])
token_counts
```




    Counter({'': 25448,
             'http': 4307,
             'number': 1890,
             'nt': 446,
             'https': 410,
             'like': 346,
             'amp': 344,
             'fire': 252,
             'get': 229,
             'new': 224,
             'via': 220,
             'people': 200,
             'news': 197,
             'one': 195,
             'us': 165,
             'video': 165,
             'emergency': 157,
             'disaster': 154,
             'would': 143,
             'police': 140,
             'still': 129,
             'body': 126,
             'got': 124,
             'california': 121,
             'crash': 120,
             'burning': 120,
             'back': 119,
             'storm': 119,
             'suicide': 116,
             'time': 112,
             'know': 112,
             'man': 111,
             'buildings': 110,
             'day': 108,
             'rt': 108,
             'first': 107,
             'see': 105,
             'world': 105,
             'ca': 103,
             'going': 103,
             'bomb': 103,
             'fires': 101,
             'nuclear': 101,
             'love': 100,
             'attack': 100,
             'today': 99,
             'two': 98,
             'youtube': 98,
             'dead': 97,
             'killed': 96,
             'go': 95,
             'train': 93,
             'gt': 92,
             'full': 91,
             'car': 90,
             'war': 90,
             'accident': 89,
             'hiroshima': 89,
             'may': 88,
             'could': 88,
             'families': 88,
             'life': 87,
             'good': 87,
             'think': 86,
             'say': 85,
             'watch': 85,
             'many': 84,
             'u': 84,
             'last': 83,
             'let': 83,
             'want': 80,
             'na': 79,
             'way': 79,
             'years': 79,
             'home': 78,
             'make': 76,
             'collapse': 75,
             'work': 74,
             'best': 74,
             'look': 73,
             'even': 73,
             'death': 73,
             'help': 72,
             'please': 72,
             'need': 72,
             'army': 72,
             'wildfire': 72,
             'mhnumber': 72,
             'another': 71,
             'really': 71,
             'take': 71,
             'lol': 71,
             'mass': 71,
             'year': 69,
             'pm': 68,
             'right': 68,
             'bombing': 68,
             'school': 67,
             'black': 67,
             'hot': 67,
             'forest': 65,
             '\x89û': 65,
             'fatal': 65,
             'much': 64,
             'northern': 64,
             'reddit': 64,
             'city': 63,
             'obama': 63,
             'great': 62,
             'water': 62,
             'never': 61,
             'homes': 61,
             'legionnaires': 61,
             'bomber': 61,
             'live': 60,
             'god': 60,
             'wreck': 60,
             'latest': 60,
             'old': 60,
             'every': 59,
             'japan': 59,
             'read': 58,
             'atomic': 58,
             'flood': 57,
             'everyone': 57,
             'said': 57,
             'flames': 57,
             'fear': 57,
             'floods': 57,
             'getting': 56,
             'shit': 56,
             'come': 56,
             'damage': 55,
             'im': 55,
             'feel': 55,
             'near': 54,
             'top': 54,
             'ever': 54,
             'content': 54,
             'hope': 53,
             'injured': 53,
             'oil': 53,
             'found': 52,
             'hit': 52,
             'since': 52,
             'weather': 52,
             'military': 52,
             'coming': 51,
             'night': 51,
             'without': 51,
             'ass': 51,
             'earthquake': 50,
             'evacuation': 50,
             'flooding': 50,
             'next': 50,
             'truck': 50,
             'stop': 50,
             'debris': 50,
             'state': 50,
             'malaysia': 50,
             'plan': 49,
             'smoke': 48,
             'set': 48,
             'times': 48,
             'little': 48,
             'cause': 48,
             'wild': 48,
             'gon': 47,
             'w': 47,
             'face': 47,
             'movie': 47,
             'wounded': 47,
             'cross': 47,
             'confirmed': 47,
             'thunderstorm': 47,
             'severe': 47,
             'heat': 46,
             'always': 46,
             'check': 46,
             'fucking': 46,
             'looks': 46,
             'well': 46,
             'food': 46,
             'numberth': 45,
             'bad': 45,
             'warning': 45,
             'weapon': 45,
             'says': 45,
             'natural': 45,
             'sinking': 45,
             'thunder': 45,
             'rain': 44,
             'also': 44,
             'bloody': 44,
             'family': 44,
             'services': 44,
             'change': 44,
             'liked': 44,
             'fall': 44,
             'lightning': 44,
             'injuries': 44,
             'loud': 44,
             'summer': 43,
             'injury': 43,
             'someone': 43,
             'free': 43,
             'screaming': 43,
             'house': 43,
             'high': 43,
             'missing': 43,
             'made': 43,
             'weapons': 43,
             'bags': 43,
             'evacuate': 43,
             'spill': 43,
             'end': 42,
             'murder': 42,
             'blood': 42,
             'run': 42,
             'boy': 42,
             'hail': 42,
             'trapped': 42,
             'collided': 42,
             'refugees': 42,
             'photo': 41,
             'head': 41,
             'tonight': 41,
             'explosion': 41,
             'air': 41,
             'destroy': 41,
             'whole': 41,
             'save': 41,
             'survive': 41,
             'released': 41,
             'attacked': 41,
             'explode': 41,
             'derailment': 41,
             'failure': 41,
             'panic': 41,
             'wreckage': 41,
             'outbreak': 41,
             'area': 40,
             'around': 40,
             'girl': 40,
             'destroyed': 40,
             'big': 40,
             'saudi': 40,
             'terrorist': 40,
             'bag': 40,
             'wind': 40,
             'hurricane': 40,
             'bridge': 40,
             'rescue': 40,
             'fatalities': 40,
             'sinkhole': 40,
             'breaking': 39,
             '\x89ûò': 39,
             'burned': 39,
             'trauma': 39,
             'ambulance': 39,
             'destruction': 39,
             'charged': 39,
             'fuck': 39,
             'story': 39,
             'post': 39,
             'keep': 39,
             'report': 39,
             'rescuers': 39,
             'lives': 39,
             'rescued': 39,
             'deaths': 39,
             'migrants': 39,
             'survived': 39,
             'wrecked': 39,
             'update': 38,
             'county': 38,
             'week': 38,
             'road': 38,
             'island': 38,
             'survivors': 38,
             'away': 38,
             'show': 38,
             'real': 38,
             'boat': 38,
             'ruin': 38,
             'service': 38,
             'catastrophe': 38,
             'harm': 38,
             'landslide': 38,
             'dust': 38,
             'twister': 38,
             'bus': 37,
             'game': 37,
             'phone': 37,
             'call': 37,
             'numberpm': 37,
             'armageddon': 37,
             'women': 37,
             'violent': 37,
             'white': 37,
             'things': 37,
             'terrorism': 37,
             'put': 37,
             'drought': 37,
             'collapsed': 37,
             'crush': 37,
             'curfew': 37,
             'danger': 37,
             'deluge': 37,
             'investigators': 37,
             'hostages': 37,
             'mudslide': 37,
             'structural': 37,
             'sandstorm': 37,
             'whirlwind': 37,
             'better': 36,
             'least': 36,
             'came': 36,
             'thing': 36,
             'airplane': 36,
             'suspect': 36,
             'crashed': 36,
             'ok': 36,
             'august': 36,
             'red': 36,
             'saw': 36,
             'woman': 36,
             'battle': 36,
             'hostage': 36,
             'tragedy': 36,
             'rioting': 36,
             'riot': 36,
             'hazard': 36,
             'hazardous': 36,
             'drowning': 36,
             'engulfed': 36,
             'screamed': 36,
             'massacre': 36,
             'bang': 36,
             'quarantine': 36,
             'quarantined': 36,
             'sunk': 36,
             'windstorm': 36,
             'horrible': 35,
             'past': 35,
             'heard': 35,
             'kills': 35,
             'iran': 35,
             'national': 35,
             'apocalypse': 35,
             'long': 35,
             'mosque': 35,
             'group': 35,
             'power': 35,
             'wan': 35,
             'oh': 35,
             'bleeding': 35,
             'anniversary': 35,
             'bombed': 35,
             'displaced': 35,
             'traumatised': 35,
             'cliff': 35,
             'china': 35,
             'stock': 35,
             'derail': 35,
             'devastation': 35,
             'exploded': 35,
             'famine': 35,
             'inundated': 35,
             'wave': 34,
             'use': 34,
             'meltdown': 34,
             'must': 34,
             'went': 34,
             'heart': 34,
             'tomorrow': 34,
             'part': 34,
             'twitter': 34,
             'fedex': 34,
             'blew': 34,
             'drown': 34,
             'blown': 34,
             'casualties': 34,
             'wounds': 34,
             'hundreds': 34,
             'desolation': 34,
             'trouble': 34,
             'detonate': 34,
             'electrocuted': 34,
             'lava': 34,
             'screams': 34,
             'something': 33,
             'inumber': 33,
             'thank': 33,
             'reunion': 33,
             'plane': 33,
             'ebay': 33,
             'lot': 33,
             'soon': 33,
             'calgary': 33,
             'bioterror': 33,
             'half': 33,
             'catastrophic': 33,
             'affected': 33,
             'chemical': 33,
             'collide': 33,
             'derailed': 33,
             'evacuated': 33,
             'flattened': 33,
             'panicking': 33,
             'left': 32,
             'possible': 32,
             'market': 32,
             'land': 32,
             'send': 32,
             'baby': 32,
             'typhoon': 32,
             'isis': 32,
             'demolish': 32,
             'fatality': 32,
             'hijacker': 32,
             'hijacking': 32,
             'pandemonium': 32,
             'razed': 32,
             'due': 31,
             'tornado': 31,
             'cool': 31,
             'care': 31,
             'traffic': 31,
             'goes': 31,
             'government': 31,
             'annihilated': 31,
             'thought': 31,
             'zone': 31,
             'kill': 31,
             'officials': 31,
             'sure': 31,
             'longer': 31,
             'security': 31,
             'blazing': 31,
             'song': 31,
             'light': 31,
             'bagging': 31,
             'pkk': 31,
             'caused': 31,
             'responders': 31,
             'collision': 31,
             'detonation': 31,
             'tsunami': 31,
             'murderer': 31,
             'obliterated': 31,
             'detonated': 31,
             'building': 30,
             'st': 30,
             'thanks': 30,
             'used': 30,
             'kids': 30,
             'issues': 30,
             'minute': 30,
             'airport': 30,
             'river': 30,
             'fan': 30,
             'arson': 30,
             'sound': 30,
             'stay': 30,
             'yet': 30,
             'india': 30,
             'beautiful': 30,
             'nothing': 30,
             'ur': 30,
             'shoulder': 30,
             'bush': 30,
             'crushed': 30,
             'blast': 30,
             'demolished': 30,
             'demolition': 30,
             'drowned': 30,
             'volcano': 30,
             'prebreak': 30,
             'obliterate': 30,
             'obliteration': 30,
             'three': 29,
             'shooting': 29,
             'already': 29,
             'making': 29,
             'done': 29,
             'men': 29,
             'believe': 29,
             'lt': 29,
             'fun': 29,
             'fight': 29,
             'start': 29,
             'remember': 29,
             'music': 29,
             'officer': 29,
             'cyclone': 29,
             'electrocute': 29,
             'eyewitness': 29,
             'hellfire': 29,
             'rainstorm': 29,
             'upheaval': 29,
             'died': 28,
             'far': 28,
             'south': 28,
             'days': 28,
             'ablaze': 28,
             'inside': 28,
             'leave': 28,
             'actually': 28,
             'wake': 28,
             'sirens': 28,
             'israeli': 28,
             'media': 28,
             'person': 28,
             'words': 28,
             'policy': 28,
             'turkey': 28,
             'hijack': 28,
             'sue': 28,
             'numberyr': 28,
             'site': 27,
             'shot': 27,
             'north': 27,
             'die': 27,
             'bc': 27,
             'hours': 27,
             'trying': 27,
             'lab': 27,
             'yes': 27,
             'abc': 27,
             're\x89û': 27,
             'nearby': 27,
             'declares': 27,
             'numberkm': 27,
             'seismic': 27,
             'numberyearold': 27,
             'snowstorm': 27,
             'place': 26,
             'wait': 26,
             'second': 26,
             '\x89ûó': 26,
             'nowplaying': 26,
             'n': 26,
             'plans': 26,
             'gets': 26,
             'brown': 26,
             'play': 26,
             'ago': 26,
             'horror': 26,
             'history': 26,
             'b': 26,
             'children': 26,
             'anything': 26,
             'guys': 26,
             'avalanche': 26,
             'health': 26,
             'pic': 26,
             'blight': 26,
             'find': 26,
             'mp': 26,
             'casualty': 26,
             'soudelor': 26,
             'deluged': 26,
             'islam': 26,
             'reactor': 26,
             'rubble': 26,
             'swallowed': 26,
             'siren': 26,
             'la': 25,
             'lost': 25,
             'outside': 25,
             'west': 25,
             'tell': 25,
             'job': 25,
             'almost': 25,
             'aircraft': 25,
             'helicopter': 25,
             'hey': 25,
             'maybe': 25,
             'peace': 25,
             'data': 25,
             'american': 25,
             'business': 25,
             'yeah': 25,
             'deal': 25,
             'bioterrorism': 25,
             'line': 25,
             'photos': 25,
             'watching': 25,
             'bigger': 25,
             'memories': 25,
             'typhoondevastated': 25,
             'saipan': 25,
             'stretcher': 25,
             'conclusively': 25,
             'street': 24,
             'america': 24,
             'support': 24,
             'book': 24,
             'anyone': 24,
             'reuters': 24,
             'pakistan': 24,
             'country': 24,
             'bar': 24,
             'hell': 24,
             'order': 24,
             'pick': 24,
             'control': 24,
             'makes': 24,
             'literally': 24,
             'transport': 24,
             'searching': 24,
             'low': 24,
             'money': 24,
             'hear': 24,
             'crews': 24,
             'rise': 24,
             'waves': 24,
             'bodies': 24,
             'projected': 24,
             'bestnaijamade': 24,
             'side': 23,
             'happy': 23,
             'center': 23,
             'finally': 23,
             'might': 23,
             'eyes': 23,
             'everything': 23,
             'wo': 23,
             'tv': 23,
             'amid': 23,
             'damn': 23,
             'team': 23,
             'feeling': 23,
             'hollywood': 23,
             'pretty': 23,
             'move': 23,
             'online': 23,
             'though': 23,
             'probably': 23,
             'saved': 23,
             'signs': 23,
             'effect': 23,
             'manslaughter': 23,
             'fast': 22,
             'mom': 22,
             'feared': 22,
             'seen': 22,
             'case': 22,
             'annihilation': 22,
             'major': 22,
             'child': 22,
             'name': 22,
             'crisis': 22,
             'leather': 22,
             'caught': 22,
             'town': 22,
             'blaze': 22,
             'okay': 22,
             'youth': 22,
             'space': 22,
             'spot': 22,
             'trains': 22,
             'desolate': 22,
             'trench': 22,
             'hat': 22,
             'refugio': 22,
             'costlier': 22,
             'miners': 22,
             'flash': 21,
             'flag': 21,
             'cars': 21,
             'huge': 21,
             'rd': 21,
             'daily': 21,
             'guy': 21,
             'wrong': 21,
             'jobs': 21,
             'omg': 21,
             'ship': 21,
             'crazy': 21,
             'hate': 21,
             'sorry': 21,
             'ball': 21,
             'self': 21,
             'stand': 21,
             'called': 21,
             'class': 21,
             'texas': 21,
             'needs': 21,
             'fukushima': 21,
             'nearly': 21,
             'morning': 21,
             'giant': 21,
             'course': 21,
             'banned': 21,
             'picking': 21,
             'offensive': 21,
             'reason': 20,
             'closed': 20,
             'heavy': 20,
             'across': 20,
             'haha': 20,
             'lord': 20,
             'hard': 20,
             'others': 20,
             'win': 20,
             'official': 20,
             'usa': 20,
             'poor': 20,
             'toddler': 20,
             'united': 20,
             'east': 20,
             'gun': 20,
             'worst': 20,
             'listen': 20,
             'anthrax': 20,
             'computers': 20,
             'dont': 20,
             'entire': 20,
             'level': 20,
             'pay': 20,
             'link': 20,
             'wow': 20,
             'meek': 20,
             'russian': 20,
             'aug': 20,
             'gbbo': 20,
             'houses': 20,
             'become': 20,
             'chance': 20,
             'friends': 20,
             'bbc': 20,
             'angry': 20,
             'cnn': 20,
             'ignition': 20,
             'knock': 20,
             'hailstorm': 20,
             'mayhem': 20,
             'myanmar': 19,
             'try': 19,
             'wanted': 19,
             'thousands': 19,
             'alone': 19,
             'climate': 19,
             'else': 19,
             'talk': 19,
             'vehicle': 19,
             'happened': 19,
             'aftershock': 19,
             'couple': 19,
             'radio': 19,
             'totally': 19,
             'blue': 19,
             'learn': 19,
             'truth': 19,
             'beach': 19,
             'reports': 19,
             'christian': 19,
             'temple': 19,
             'view': 19,
             'star': 19,
             'taken': 19,
             'playing': 19,
             'australia': 19,
             'mishaps': 19,
             'action': 19,
             'public': 19,
             'ai': 19,
             'running': 19,
             'looking': 19,
             'mad': 19,
             'cake': 19,
             'pain': 19,
             'blizzard': 19,
             'ladies': 19,
             'drake': 19,
             'appears': 19,
             'centre': 19,
             'village': 19,
             'takes': 19,
             'issued': 19,
             'emmerdale': 19,
             'declaration': 19,
             'disea': 19,
             'arsonist': 18,
             'front': 18,
             'property': 18,
             'drive': 18,
             'global': 18,
             'experts': 18,
             'trust': 18,
             'ready': 18,
             'vs': 18,
             'human': 18,
             'film': 18,
             'till': 18,
             'friend': 18,
             'green': 18,
             'muslims': 18,
             'mount': 18,
             'favorite': 18,
             'eye': 18,
             'germs': 18,
             'follow': 18,
             'large': 18,
             'womens': 18,
             'marks': 18,
             'thursday': 18,
             'downtown': 18,
             'insurance': 18,
             'mph': 18,
             'instead': 18,
             'coaches': 18,
             'flight': 18,
             'quiz': 18,
             'devastated': 18,
             'virgin': 18,
             'chile': 18,
             'bring': 17,
             'sky': 17,
             'taking': 17,
             'r': 17,
             'dies': 17,
             'moment': 17,
             'behind': 17,
             'four': 17,
             '\x89ûï': 17,
             'dog': 17,
             'driver': 17,
             'israel': 17,
             'numbers': 17,
             'park': 17,
             'sign': 17,
             'following': 17,
             'disease': 17,
             'comes': 17,
             'scared': 17,
             'escape': 17,
             'govt': 17,
             'date': 17,
             'hiring': 17,
             'true': 17,
             'theater': 17,
             'gave': 17,
             'gop': 17,
             'driving': 17,
             'added': 17,
             'party': 17,
             'turn': 17,
             'libya': 17,
             'nagasaki': 17,
             'outrage': 17,
             'camp': 17,
             'x': 17,
             'sounds': 17,
             'ppl': 17,
             'british': 17,
             'landing': 17,
             'download': 17,
             'york': 17,
             'patience': 17,
             'former': 17,
             'madhya': 17,
             'pradesh': 17,
             'wonder': 17,
             'numberw': 17,
             'led': 17,
             'gems': 17,
             'funtenna': 17,
             'ancient': 17,
             'subreddits': 17,
             'galactic': 17,
             'colorado': 16,
             'awesome': 16,
             'ave': 16,
             'upon': 16,
             'shots': 16,
             'reported': 16,
             'risk': 16,
             'turned': 16,
             'seeing': 16,
             'thinking': 16,
             'france': 16,
             'wednesday': 16,
             'mode': 16,
             'early': 16,
             'pakistani': 16,
             'potus': 16,
             'dad': 16,
             'give': 16,
             'bed': 16,
             'hand': 16,
             'earth': 16,
             'working': 16,
             'russia': 16,
             'mop': 16,
             'shows': 16,
             'lmao': 16,
             'claims': 16,
             'militants': 16,
             'bombs': 16,
             'pamela': 16,
             'miss': 16,
             'ebola': 16,
             'niggas': 16,
             'tweet': 16,
             'enough': 16,
             'told': 16,
             'numberst': 16,
             'soul': 16,
             'info': 16,
             'rock': 16,
             'biggest': 16,
             'sad': 16,
             'fashion': 16,
             'middle': 16,
             'pray': 16,
             'holding': 16,
             'likely': 16,
             'businesses': 16,
             'room': 16,
             'uk': 16,
             'tree': 16,
             'rules': 16,
             'lamp': 16,
             'nws': 16,
             'bayelsa': 16,
             'nigerian': 16,
             'parole': 16,
             'unconfirmed': 16,
             'neighbour': 16,
             'rly': 16,
             'expected': 15,
             'london': 15,
             'secret': 15,
             'scene': 15,
             'guess': 15,
             'wants': 15,
             'tried': 15,
             'members': 15,
             'safety': 15,
             'share': 15,
             'started': 15,
             'series': 15,
             'department': 15,
             'japanese': 15,
             'arrested': 15,
             'terror': 15,
             'shift': 15,
             'waving': 15,
             'geller': 15,
             'playlist': 15,
             'king': 15,
             'young': 15,
             'research': 15,
             'cut': 15,
             'problem': 15,
             'ahead': 15,
             'dude': 15,
             'p': 15,
             'super': 15,
             'tote': 15,
             'handbag': 15,
             'point': 15,
             'investigating': 15,
             'washington': 15,
             'coast': 15,
             'worse': 15,
             'sex': 15,
             'don\x89ûªt': 15,
             'gas': 15,
             'fans': 15,
             'sea': 15,
             'disney': 15,
             'interesting': 15,
             'passengers': 15,
             'alarm': 15,
             'break': 15,
             'apollo': 15,
             'cree': 15,
             'sick': 15,
             'cdt': 15,
             'lake': 14,
             'aba': 14,
             'season': 14,
             'means': 14,
             'teen': 14,
             'dance': 14,
             'twelve': 14,
             'petition': 14,
             'prepare': 14,
             'direction': 14,
             'indian': 14,
             'victims': 14,
             'cant': 14,
             'internet': 14,
             'act': 14,
             'album': 14,
             'general': 14,
             ...})




```python
token_counts=Counter([token for sublist in list(train_data['text']) for token in sublist])
vocab={word:i+1 for i, (word,_) in enumerate(token_counts.items())}
vocab
```




    {'deeds': 1,
     'reason': 2,
     '': 3,
     'earthquake': 4,
     'may': 5,
     'allah': 6,
     'forgive': 7,
     'us': 8,
     'forest': 9,
     'fire': 10,
     'near': 11,
     'la': 12,
     'ronge': 13,
     'sask': 14,
     'canada': 15,
     'residents': 16,
     'asked': 17,
     'shelter': 18,
     'place': 19,
     'notified': 20,
     'officers': 21,
     'evacuation': 22,
     'orders': 23,
     'expected': 24,
     'number': 25,
     'people': 26,
     'receive': 27,
     'wildfires': 28,
     'california': 29,
     'got': 30,
     'sent': 31,
     'photo': 32,
     'ruby': 33,
     'alaska': 34,
     'smoke': 35,
     'pours': 36,
     'school': 37,
     'rockyfire': 38,
     'update': 39,
     'hwy': 40,
     'closed': 41,
     'directions': 42,
     'due': 43,
     'lake': 44,
     'county': 45,
     'cafire': 46,
     'flood': 47,
     'disaster': 48,
     'heavy': 49,
     'rain': 50,
     'causes': 51,
     'flash': 52,
     'flooding': 53,
     'streets': 54,
     'manitou': 55,
     'colorado': 56,
     'springs': 57,
     'areas': 58,
     'top': 59,
     'hill': 60,
     'see': 61,
     'woods': 62,
     'emergency': 63,
     'happening': 64,
     'building': 65,
     'across': 66,
     'street': 67,
     'afraid': 68,
     'tornado': 69,
     'coming': 70,
     'area': 71,
     'three': 72,
     'died': 73,
     'heat': 74,
     'wave': 75,
     'far': 76,
     'haha': 77,
     'south': 78,
     'tampa': 79,
     'getting': 80,
     'flooded': 81,
     'hah': 82,
     'wait': 83,
     'second': 84,
     'live': 85,
     'gon': 86,
     'na': 87,
     'fvck': 88,
     'raining': 89,
     'florida': 90,
     'tampabay': 91,
     'days': 92,
     'lost': 93,
     'count': 94,
     'bago': 95,
     'myanmar': 96,
     'arrived': 97,
     'damage': 98,
     'bus': 99,
     'multi': 100,
     'car': 101,
     'crash': 102,
     'breaking': 103,
     'man': 104,
     'love': 105,
     'fruits': 106,
     'summer': 107,
     'lovely': 108,
     'fast': 109,
     'goooooooaaaaaal': 110,
     'ridiculous': 111,
     'london': 112,
     'cool': 113,
     'skiing': 114,
     'wonderful': 115,
     'day': 116,
     'looooool': 117,
     'way': 118,
     'ca': 119,
     'nt': 120,
     'eat': 121,
     'shit': 122,
     'nyc': 123,
     'last': 124,
     'week': 125,
     'girlfriend': 126,
     'cooool': 127,
     'like': 128,
     'pasta': 129,
     'end': 130,
     'bbcmtd': 131,
     'wholesale': 132,
     'markets': 133,
     'ablaze': 134,
     'http': 135,
     'tcolhyxeohynumberc': 136,
     'always': 137,
     'try': 138,
     'bring': 139,
     'metal': 140,
     'rt': 141,
     'tcoyaonumberenumberxngw': 142,
     'africanbaze': 143,
     'news': 144,
     'nigeria': 145,
     'flag': 146,
     'set': 147,
     'aba': 148,
     'tconumbernndbgwyei': 149,
     'crying': 150,
     'plus': 151,
     'side': 152,
     'look': 153,
     'sky': 154,
     'night': 155,
     'tcoqqsmshajnumbern': 156,
     'phdsquares': 157,
     'mufc': 158,
     'built': 159,
     'much': 160,
     'hype': 161,
     'around': 162,
     'new': 163,
     'acquisitions': 164,
     'doubt': 165,
     'epl': 166,
     'season': 167,
     'inec': 168,
     'office': 169,
     'abia': 170,
     'tconumberimaomknna': 171,
     'barbados': 172,
     'bridgetown': 173,
     'jamaica': 174,
     '\x89ûò': 175,
     'two': 176,
     'cars': 177,
     'santa': 178,
     'cruz': 179,
     '\x89ûó': 180,
     'head': 181,
     'st': 182,
     'elizabeth': 183,
     'police': 184,
     'superintende': 185,
     'tcowdueajnumberqnumberj': 186,
     'lord': 187,
     'check': 188,
     'tcoroinumbernsmejj': 189,
     'tconumbertjnumberzjinnumber': 190,
     'tcoyduixefipe': 191,
     'tcolxtjcnumberkls': 192,
     'nsfw': 193,
     'outside': 194,
     'alive': 195,
     'dead': 196,
     'inside': 197,
     'awesome': 198,
     'time': 199,
     'visiting': 200,
     'cfc': 201,
     'ancop': 202,
     'site': 203,
     'thanks': 204,
     'tita': 205,
     'vida': 206,
     'taking': 207,
     'care': 208,
     'soooo': 209,
     'pumped': 210,
     'southridgelife': 211,
     'wanted': 212,
     'chicago': 213,
     'preaching': 214,
     'hotel': 215,
     'tcoonumberqknbfofx': 216,
     'gained': 217,
     'followers': 218,
     'know': 219,
     'stats': 220,
     'grow': 221,
     'tcotiyulifnumbercnumber': 222,
     'west': 223,
     'burned': 224,
     'thousands': 225,
     'alone': 226,
     'tcovlnumbertbrnumberwbr': 227,
     'perfect': 228,
     'tracklist': 229,
     'life': 230,
     'leave': 231,
     'first': 232,
     'retainers': 233,
     'quite': 234,
     'weird': 235,
     'better': 236,
     'get': 237,
     'used': 238,
     'wear': 239,
     'every': 240,
     'single': 241,
     'next': 242,
     'year': 243,
     'least': 244,
     'deputies': 245,
     'shot': 246,
     'brighton': 247,
     'home': 248,
     'tcogwnrhmsonumberk': 249,
     'wife': 250,
     'six': 251,
     'years': 252,
     'jail': 253,
     'setting': 254,
     'niece': 255,
     'tcoevnumberahoucza': 256,
     'superintendent': 257,
     'lanford': 258,
     'salmon': 259,
     'r': 260,
     'tcovplrnumberhkanumberu': 261,
     'tcosxhwnumbertnnlf': 262,
     'arsonist': 263,
     'deliberately': 264,
     'black': 265,
     'church': 266,
     'north': 267,
     'carolinaåêablaze': 268,
     'tcopcxarbhnumberan': 269,
     'noches': 270,
     'elbestia': 271,
     'alexissanchez': 272,
     'happy': 273,
     'teammates': 274,
     'training': 275,
     'hard': 276,
     'goodnight': 277,
     'gunners': 278,
     'tcoucnumberjnumberjhvgr': 279,
     'kurds': 280,
     'trampling': 281,
     'turkmen': 282,
     'later': 283,
     'others': 284,
     'vandalized': 285,
     'offices': 286,
     'front': 287,
     'diyala': 288,
     'tconumberizfdycnumbercg': 289,
     'truck': 290,
     'rnumber': 291,
     'voortrekker': 292,
     'ave': 293,
     'tambo': 294,
     'intl': 295,
     'cargo': 296,
     'section': 297,
     'tconumberkscqkfkkf': 298,
     'hearts': 299,
     'city': 300,
     'gift': 301,
     'skyline': 302,
     'kiss': 303,
     'upon': 304,
     'lips': 305,
     '\x89û': 306,
     'https': 307,
     'tcocyompznumberanumberz': 308,
     'tonight': 309,
     'los': 310,
     'angeles': 311,
     'expecting': 312,
     'ig': 313,
     'fb': 314,
     'filled': 315,
     'sunset': 316,
     'shots': 317,
     'peeps': 318,
     'tcoicsjgznumbertenumber': 319,
     'climate': 320,
     'energy': 321,
     'tconumberfxmnnumberlnumberbd': 322,
     'revel': 323,
     'wmv': 324,
     'videos': 325,
     'means': 326,
     'mac': 327,
     'farewell': 328,
     'en': 329,
     'route': 330,
     'dvd': 331,
     'gtxrwm': 332,
     'progressive': 333,
     'greetings': 334,
     'month': 335,
     'students': 336,
     'would': 337,
     'pens': 338,
     'torch': 339,
     'publications': 340,
     'tconumberfxpixqujt': 341,
     'rene': 342,
     'amp': 343,
     'jacinta': 344,
     'secret': 345,
     'numberknumber': 346,
     'fallen': 347,
     'skies': 348,
     'edit': 349,
     'mar': 350,
     'tconumbermlmsuzvnumberz': 351,
     'navistanumber': 352,
     'steve': 353,
     'fires': 354,
     'something': 355,
     'else': 356,
     'tinderbox': 357,
     'clown': 358,
     'hood': 359,
     'newsnumber': 360,
     'nowplaying': 361,
     'ian': 362,
     'buff': 363,
     'magnitude': 364,
     'tcoavnumberjsjfftc': 365,
     'edm': 366,
     'nxwestmidlands': 367,
     'huge': 368,
     'tcorwzbfvnxer': 369,
     'talk': 370,
     'go': 371,
     'make': 372,
     'work': 373,
     'kids': 374,
     'cuz': 375,
     'bicycle': 376,
     'accident': 377,
     'split': 378,
     'testicles': 379,
     'impossible': 380,
     'michael': 381,
     'father': 382,
     'inumber': 383,
     'w': 384,
     'nashvilletraffic': 385,
     'traffic': 386,
     'moving': 387,
     'numberm': 388,
     'slower': 389,
     'usual': 390,
     'tconumberghknumberegj': 391,
     'center': 392,
     'lane': 393,
     'blocked': 394,
     'santaclara': 395,
     'usnumber': 396,
     'nb': 397,
     'great': 398,
     'america': 399,
     'pkwy': 400,
     'bayarea': 401,
     'tcopmlohzurwr': 402,
     'tcogkyenumbergjtknumber': 403,
     'personalinjury': 404,
     'read': 405,
     'advice': 406,
     'solicitor': 407,
     'help': 408,
     'otleyhour': 409,
     'stlouis': 410,
     'caraccidentlawyer': 411,
     'speeding': 412,
     'among': 413,
     'teen': 414,
     'accidents': 415,
     'tcoknumberzomofnumber': 416,
     'tcosnumberkxvmnumbercba': 417,
     'tee\x89û': 418,
     'reported': 419,
     'motor': 420,
     'vehicle': 421,
     'curry': 422,
     'herman': 423,
     'rd': 424,
     'stephenson': 425,
     'involving': 426,
     'overturned': 427,
     'please': 428,
     'use': 429,
     'tcoybjezkurwnumber': 430,
     'bigrigradio': 431,
     'awareness': 432,
     'mile': 433,
     'marker': 434,
     'mooresville': 435,
     'iredell': 436,
     'ramp': 437,
     'pm': 438,
     'sleepjunkies': 439,
     'sleeping': 440,
     'pills': 441,
     'double': 442,
     'risk': 443,
     'tconumbersnumbernmnumberfict': 444,
     'knew': 445,
     'happen': 446,
     'tcoysxunnumbervceh': 447,
     'n': 448,
     'cabrillo': 449,
     'hwymagellan': 450,
     'av': 451,
     'mir': 452,
     'congestion': 453,
     'pastor': 454,
     'scene': 455,
     'owner': 456,
     'range': 457,
     'rover': 458,
     'mom': 459,
     'wished': 460,
     'spilt': 461,
     'mayonnaise': 462,
     'horrible': 463,
     'past': 464,
     'sunday': 465,
     'finally': 466,
     'able': 467,
     'thank': 468,
     'god': 469,
     'pissed': 470,
     'donnie': 471,
     'tell': 472,
     'another': 473,
     'truckcrash': 474,
     'overturns': 475,
     'fortworth': 476,
     'interstate': 477,
     'tcorsnumberljnumberqfp': 478,
     'click': 479,
     'gt': 480,
     'tcoldnumberuniywnumberk': 481,
     'ashville': 482,
     'sb': 483,
     'sr': 484,
     'tcohylmonumberwgfi': 485,
     'carolina': 486,
     'motorcyclist': 487,
     'dies': 488,
     'crossed': 489,
     'median': 490,
     'motorcycle': 491,
     'rider': 492,
     'traveling': 493,
     'tcopnumberlzrlmynumber': 494,
     'fyi': 495,
     'cad': 496,
     'property': 497,
     'nhs': 498,
     'piner': 499,
     'rdhorndale': 500,
     'dr': 501,
     'naayf': 502,
     'turning': 503,
     'onto': 504,
     'chandanee': 505,
     'magu': 506,
     'mma': 507,
     'taxi': 508,
     'rammed': 509,
     'halfway': 510,
     'turned': 511,
     'everyone': 512,
     'conf\x89û': 513,
     'left': 514,
     'manchester': 515,
     'eddy': 516,
     'stop': 517,
     'back': 518,
     'nhnumbera': 519,
     'delay': 520,
     'mins': 521,
     'tcooianumberfxinumbergm': 522,
     'wpd': 523,
     'numberth': 524,
     'injury': 525,
     'willis': 526,
     'foreman': 527,
     'tcovckitnumberedev': 528,
     'aashiqui': 529,
     'actress': 530,
     'anu': 531,
     'aggarwal': 532,
     'nearfatal': 533,
     'tconumberotfpnumberlqw': 534,
     'suffield': 535,
     'alberta': 536,
     'tcobptmlfnumberpnumber': 537,
     'backup': 538,
     'blocking': 539,
     'right': 540,
     'lanes': 541,
     'exit': 542,
     'langtree': 543,
     'consider': 544,
     'nc': 545,
     'alternate': 546,
     'changed': 547,
     'determine': 548,
     'options': 549,
     'financially': 550,
     'support': 551,
     'plans': 552,
     'ongoing': 553,
     'treatment': 554,
     'deadly': 555,
     'happened': 556,
     'hagerstown': 557,
     'today': 558,
     'details': 559,
     'yournumberstate': 560,
     'whag': 561,
     'flowri': 562,
     'marinading': 563,
     'even': 564,
     'fucking': 565,
     'mfs': 566,
     'drive': 567,
     'norwaymfa': 568,
     'bahrain': 569,
     'previously': 570,
     'road': 571,
     'killed': 572,
     'explosion': 573,
     'tcogfjfgtodad': 574,
     'still': 575,
     'heard': 576,
     'leaders': 577,
     'kenya': 578,
     'forward': 579,
     'comment': 580,
     'issue': 581,
     'disciplinary': 582,
     'measures': 583,
     'arrestpastornganga': 584,
     'aftershockdelo': 585,
     'scuf': 586,
     'ps': 587,
     'game': 588,
     'cya': 589,
     'effort': 590,
     'gets': 591,
     'painful': 592,
     'win': 593,
     'roger': 594,
     'bannister': 595,
     'ir': 596,
     'icemoon': 597,
     'aftershock': 598,
     'tcoynxnvvkcda': 599,
     'djicemoon': 600,
     'dubstep': 601,
     'trapmusic': 602,
     'dnb': 603,
     'dance': 604,
     'ices\x89û': 605,
     'tcoweqpesenku': 606,
     'victory': 607,
     'bargain': 608,
     'basement': 609,
     'prices': 610,
     'dwight': 611,
     'david': 612,
     'eisenhower': 613,
     'tcovamnumberpodgyw': 614,
     'tcozevakjapcz': 615,
     'nobody': 616,
     'remembers': 617,
     'came': 618,
     'charles': 619,
     'schulz': 620,
     'im': 621,
     'speaking': 622,
     'someone': 623,
     'using': 624,
     'xbnumber': 625,
     'also': 626,
     'harder': 627,
     'conflict': 628,
     'glorious': 629,
     'triumph': 630,
     'thomas': 631,
     'paine': 632,
     'growingupspoiled': 633,
     'going': 634,
     'clay': 635,
     'pigeon': 636,
     'shooting': 637,
     'guess': 638,
     'one': 639,
     'actually': 640,
     'wants': 641,
     'free': 642,
     'tc': 643,
     'terrifying': 644,
     'best': 645,
     'roller': 646,
     'coaster': 647,
     'ever': 648,
     'disclaimer': 649,
     'tcoxmwodfmtui': 650,
     'tcomnumberjdzmgjow': 651,
     'tconnumberuhasfkbv': 652,
     'tcoenumberepzhoth': 653,
     'tconumberanumberdnumberdonumberq': 654,
     'kjfordays': 655,
     'seeing': 656,
     'issues': 657,
     'tcothyzomvwunumber': 658,
     'tconumberjoonumberxknumber': 659,
     'wisdomwed': 660,
     'bonus': 661,
     'minute': 662,
     'daily': 663,
     'habits': 664,
     'could': 665,
     'really': 666,
     'improve': 667,
     'many': 668,
     'already': 669,
     'lifehacks': 670,
     'tcotbmnumberfqbnumbercw': 671,
     'protect': 672,
     'profit': 673,
     'global': 674,
     'financial': 675,
     'meltdown': 676,
     'wiedemer': 677,
     'tcowztznumberhgmvq': 678,
     'moment': 679,
     'scary': 680,
     'guy': 681,
     'behind': 682,
     'screaming': 683,
     'bloody': 684,
     'murder': 685,
     'silverwood': 686,
     '\x89ã¢': 687,
     'full\x89ã¢': 688,
     'streaming': 689,
     'youtube': 690,
     'tcovvenumberusesgf': 691,
     'book': 692,
     'tcofnumberntucnumberz': 693,
     'esquireattire': 694,
     'sometimes': 695,
     'face': 696,
     'difficulties': 697,
     'wrong': 698,
     'joel': 699,
     'osteen': 700,
     'thing': 701,
     'stands': 702,
     'dream': 703,
     'belief': 704,
     'possible': 705,
     'brown': 706,
     'praise': 707,
     'ministry': 708,
     'tells': 709,
     'wdyouth': 710,
     'biblestudy': 711,
     'tcoujknumberenumbergbcc': 712,
     'remembering': 713,
     'die': 714,
     'avoid': 715,
     'trap': 716,
     'thinking': 717,
     'lose': 718,
     'jobs': 719,
     'tried': 720,
     'orange': 721,
     'never': 722,
     'onfireanders': 723,
     'bb': 724,
     'tcojvnumberppkhjynumber': 725,
     'kick': 726,
     'want': 727,
     'making': 728,
     'say': 729,
     'done': 730,
     'interrupt': 731,
     'george': 732,
     'bernard': 733,
     'shaw': 734,
     'oyster': 735,
     'shell': 736,
     'andrew': 737,
     'carnegie': 738,
     'anyone': 739,
     'need': 740,
     'pu': 741,
     'play': 742,
     'hybrid': 743,
     'slayer': 744,
     'psnumber': 745,
     'eu': 746,
     'hmu': 747,
     'codnumbersandscrims': 748,
     'empirikgaming': 749,
     'codawscrims': 750,
     'numbertpkotc': 751,
     'numbertpfa': 752,
     'aftershockorg': 753,
     'experts': 754,
     'france': 755,
     'begin': 756,
     'examining': 757,
     'airplane': 758,
     'debris': 759,
     'found': 760,
     'reunion': 761,
     'island': 762,
     'french': 763,
     'air': 764,
     'tcoyvvpznzmxg': 765,
     'strict': 766,
     'liability': 767,
     'context': 768,
     'pilot': 769,
     'error': 770,
     'common': 771,
     'component': 772,
     'aviation': 773,
     'cr': 774,
     'tconumbercznumberbohrdnumber': 775,
     'crobscarla': 776,
     'lifetime': 777,
     'odds': 778,
     'dying': 779,
     'wedn': 780,
     'tcobkpfpogysi': 781,
     'alexalltimelow': 782,
     'awwww': 783,
     'cuties': 784,
     'good': 785,
     'job': 786,
     'family': 787,
     'members': 788,
     'osama': 789,
     'bin': 790,
     'laden': 791,
     'ironic': 792,
     'mhmmm': 793,
     'gov': 794,
     'suspect': 795,
     'goes': 796,
     'engine': 797,
     'tcotyjxrfdnumberst': 798,
     'via': 799,
     'wings': 800,
     'tcoinumberkztevbnumberv': 801,
     'cessna': 802,
     'ocampo': 803,
     'coahuila': 804,
     'mexico': 805,
     'july': 806,
     'four': 807,
     'men': 808,
     'including': 809,
     'state': 810,
     'government': 811,
     'official': 812,
     'watchthevideo': 813,
     'tcopnumberxrvgjik': 814,
     'tcolsmxnumbervwrnumberj': 815,
     'wednesday\x89û': 816,
     'wednesday': 817,
     'began': 818,
     'kca': 819,
     'votejktnumberid': 820,
     'mbataweel': 821,
     'rip': 822,
     'binladen': 823,
     'almost': 824,
     'coworker': 825,
     'nudes': 826,
     'mode': 827,
     'mickinyman': 828,
     'theatlantic': 829,
     'might': 830,
     'wreck': 831,
     'politics': 832,
     'tcotagzbcxfjnumber': 833,
     'mlb': 834,
     'unbelievably': 835,
     'insane': 836,
     'airport': 837,
     'aircraft': 838,
     'aeroplane': 839,
     'runway': 840,
     'freaky\x89û': 841,
     'tcocezhqnumberczll': 842,
     'airplaneåê': 843,
     'tcowqnumberwjsgphl': 844,
     'tcotfcdronranumber': 845,
     'usama': 846,
     'ladins': 847,
     'naturally': 848,
     'plane': 849,
     'festival': 850,
     'tcokqnumberaenumberapnumberb': 851,
     'death': 852,
     'carfest': 853,
     'tcogibyqhhkpk': 854,
     'dtn': 855,
     'brazil': 856,
     'exp': 857,
     'tcomnumberignumberwqnumberlq': 858,
     'tcovnumbersmaeslknumber': 859,
     '\x89ûïairplane\x89û\x9d': 860,
     'wtf': 861,
     'can\x89ûªt': 862,
     'believe': 863,
     'eyes': 864,
     'tconumberffylajwps': 865,
     'nicole': 866,
     'fletcher': 867,
     'victim': 868,
     'crashed': 869,
     'times': 870,
     'ago': 871,
     'little': 872,
     'bit': 873,
     'trauma': 874,
     'although': 875,
     'omg': 876,
     'tcoxdxdprcpns': 877,
     'bro': 878,
     'jetengine': 879,
     'turbojet': 880,
     'boing': 881,
     'gnumber': 882,
     'tcokxxnszpnumbernk': 883,
     'phone': 884,
     'looks': 885,
     'ship': 886,
     'terrible': 887,
     'statistically': 888,
     'cop': 889,
     'crashes': 890,
     'house': 891,
     'colombia': 892,
     'tcozhjlflbhzl': 893,
     'tcoieccnumberjdoub': 894,
     'drone': 895,
     'cause': 896,
     'pilots': 897,
     'worried': 898,
     'drones': 899,
     'esp': 900,
     'close': 901,
     'vicinity': 902,
     'airports': 903,
     'tcokznumberrgngjf': 904,
     'early': 905,
     'wake': 906,
     'call': 907,
     'sister': 908,
     'begging': 909,
     'come': 910,
     'ride': 911,
     'wher': 912,
     'ambulance': 913,
     'hospital': 914,
     'rodkiai': 915,
     'tcoaynumberzzcupnz': 916,
     'twelve': 917,
     'feared': 918,
     'pakistani': 919,
     'helicopter': 920,
     'tcoscnumberdnsnumbermc': 921,
     'ambulances': 922,
     'serious': 923,
     'lorry': 924,
     'tconumberpfeaqeski': 925,
     'tcofntgnumberrnkx': 926,
     'emsne\x89û': 927,
     'reuters': 928,
     'tcomdnugvubwn': 929,
     'yugvani': 930,
     'leading': 931,
     'services': 932,
     'boss': 933,
     'welcomes': 934,
     'charity': 935,
     'tcomjnumberjqnumberpsvnumber': 936,
     'travelling': 937,
     'aberystwythshrewsbury': 938,
     'incident': 939,
     'halt': 940,
     'shrews': 941,
     'tcoxumnumberylcbnumberq': 942,
     'sprinter': 943,
     'automatic': 944,
     'frontline': 945,
     'choice': 946,
     'lez': 947,
     'compliant': 948,
     'ebay': 949,
     'tconumberevttqpeia': 950,
     'nanotech': 951,
     'device': 952,
     'target': 953,
     'destroy': 954,
     'blood': 955,
     'clots': 956,
     'tcohfynumbervnumberslbb': 957,
     'numberskyhawkmmnumber': 958,
     'traplordnumber': 959,
     'fredosantananumber': 960,
     'lilreesenumber': 961,
     'hella': 962,
     'crazy': 963,
     'fights': 964,
     'couple': 965,
     'mosh': 966,
     'pits': 967,
     'run': 968,
     'lucky': 969,
     'justsaying': 970,
     'randomthought': 971,
     'tcobfesnumbertwbzt': 972,
     'tilnow': 973,
     'dna': 974,
     'tconumberxglahnumberzl': 975,
     'tcothmblaatzp': 976,
     'tanslash': 977,
     'waiting': 978,
     'fouseytube': 979,
     'ok': 980,
     'hahahah': 981,
     'tcozsberqnnnumbern': 982,
     'tcoqnumberivrzojzv': 983,
     'pakistan': 984,
     'kills': 985,
     'nine': 986,
     'tconumberenumberrynumberebmf': 987,
     'thenissonian': 988,
     'rejectdcartoons': 989,
     'nissan': 990,
     'medical': 991,
     'assistance': 992,
     'emsnumber': 993,
     'ny': 994,
     'emts': 995,
     'petition': 996,
     'per': 997,
     'hour': 998,
     '\x89û÷minimum': 999,
     'wage\x89ûª': 1000,
     ...}



Since there are no labels on the test data of this task, in order to evaluate the model and adjust the hyperparameters, my idea is to randomly take 20% of the training data set as the test set.


```python
#Method one, step by step, easier to understand
from torch.utils.data import DataLoader,TensorDataset

#Extract IDs and labels from train_data
train_ids = train_data['id'].values
train_labels = train_data['target'].values

#Shuffle the indices while keeping correspondence intact
total_samples = len(train_padded)
indices = np.random.permutation(total_samples)

#Shuffle the data, IDs, and labels using the shuffled indices
train_padded_shuffled = train_padded[indices]
train_labels_shuffled = train_labels[indices]
train_ids_shuffled = train_ids[indices]

# Split the shuffled data for training and testing
test_size = int(0.2 * total_samples)
train_padded_split = train_padded_shuffled[test_size:]
train_labels_split = train_labels_shuffled[test_size:]
train_ids_split = train_ids_shuffled[test_size:]

test_padded_split = train_padded_shuffled[:test_size]
test_labels_split=train_labels_shuffled[:test_size]
test_ids_split = train_ids_shuffled[:test_size]

#Convert data to PyTorch tensors
train_padded_tensor = torch.tensor(train_padded_split, dtype=torch.long)
test_padded_tensor = torch.tensor(test_padded_split, dtype=torch.long)
train_labels_tensor = torch.tensor(train_labels_split, dtype=torch.float)
test_labels_tensor=torch.tensor(test_labels_split,dtype=torch.float)

#Create DataLoader for training and testing data
train_dataset = TensorDataset(train_padded_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#Create a separate DataLoader for the test data with IDs
test_dataset = TensorDataset(test_padded_tensor,test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Method two,more common
# from sklearn.model_selection import train_test_split
# import torch
# from torch.utils.data import Dataset, DataLoader

# # Assuming train_padded, train_labels, and train_ids have been defined
# # Splitting the data while preserving IDs
# X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
#     train_padded, train_labels, train_ids, test_size=0.2, random_state=42
# )

# # Creating custom dataset class to include IDs
# class CustomDataset(Dataset):
#     def __init__(self, data, labels, ids):
#         self.data = data
#         self.labels = labels
#         self.ids = ids

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return {
#             "data": torch.tensor(self.data[idx], dtype=torch.long),
#             "label": torch.tensor(self.labels[idx], dtype=torch.float),
#             "id": self.ids[idx]
#         }

# # Instantiate CustomDataset for both training and testing
# train_dataset = CustomDataset(X_train, y_train, ids_train)
# test_dataset = CustomDataset(X_test, y_test, ids_test)

# # Wrap datasets in DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

## Train model ##

![%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20240519160910.jpg](%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20240519160910.jpg)
<center>Model architecture</center>


```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.fc2= nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.85)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.mean(1)  # Assume averaging the embeddings is the feature
        hidden = self.dropout(F.relu(self.fc(embedded)))
        output = self.fc2(hidden)
        output=self.fc3(output)
        return output
    
# Hyperparameters
vocab_size = len(vocab)+1  # Size of your vocabulary
embedding_dim = 256
hidden_dim = 256
output_dim = 1  # Binary classification
learning_rate = 0.001
batch_size = 32
num_epochs = 10

model = TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
```


```python
# Training loop
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    model.train()
    
    for text, labels in iterator:
        text=text.to(device)
        labels=labels.to(device).unsqueeze(1)
        
        outputs=model(text)
        loss=criterion(outputs,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for text, labels in iterator:
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)


for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    valid_loss = evaluate(model, test_loader, criterion)
    
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')

#Evaluation with F1 score
from sklearn.metrics import f1_score

def evaluate_f1_score(model, iterator):
    all_predictions = []
    all_labels = []
    model.eval()
    
    with torch.no_grad():
        for text, labels in iterator:
            predictions = torch.sigmoid(model(text)).squeeze(1)
            rounded_predictions = torch.round(predictions)
            all_predictions.extend(rounded_predictions.tolist())
            all_labels.extend(labels.tolist())
    
    return f1_score(all_labels, all_predictions)

f1 = evaluate_f1_score(model, test_loader)
print(f'F1 Score: {f1}')
```

    C:\Users\86135\AppData\Local\Temp\ipykernel_5304\1618821494.py:121: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      train_padded_tensor = torch.tensor(train_padded_split, dtype=torch.long)
    C:\Users\86135\AppData\Local\Temp\ipykernel_5304\1618821494.py:122: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      test_padded_tensor = torch.tensor(test_padded_split, dtype=torch.long)
    

    Epoch: 01
    	Train Loss: 0.686
    	 Val. Loss: 0.668
    Epoch: 02
    	Train Loss: 0.673
    	 Val. Loss: 0.653
    Epoch: 03
    	Train Loss: 0.649
    	 Val. Loss: 0.625
    Epoch: 04
    	Train Loss: 0.591
    	 Val. Loss: 0.559
    Epoch: 05
    	Train Loss: 0.509
    	 Val. Loss: 0.512
    Epoch: 06
    	Train Loss: 0.445
    	 Val. Loss: 0.510
    Epoch: 07
    	Train Loss: 0.379
    	 Val. Loss: 0.498
    Epoch: 08
    	Train Loss: 0.320
    	 Val. Loss: 0.554
    Epoch: 09
    	Train Loss: 0.271
    	 Val. Loss: 0.568
    Epoch: 10
    	Train Loss: 0.236
    	 Val. Loss: 0.608
    F1 Score: 0.7159468438538206
    

## Genenate the submission file ##


```python
# Create a DataLoader for your test data
test_loader = DataLoader(test_padded, batch_size=32, shuffle=False)

def predict(model, iterator):
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():  # No need to track gradients for predictions
        for text in iterator:
            text = text.to(device)
            output = model(text)  # Forward pass, get logits
            probability = torch.sigmoid(output)  # Apply sigmoid to get probabilities
            predicted_labels = torch.round(probability).squeeze()  # Convert probabilities to 0 or 1
            predictions.extend(predicted_labels.tolist())  # Extend the flat list with new sublist

    return predictions

# Use the trained model to predict labels for test data
test_predictions = predict(model, test_loader)

# Convert predictions to integers
test_predictions = [int(pred) for pred in test_predictions]

sample_submission = pd.read_csv("D:/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission['target'] = test_predictions

sample_submission.to_csv('final_submission.csv', index=False)
```
