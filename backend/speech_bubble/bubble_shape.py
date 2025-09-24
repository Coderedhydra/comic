from transformers import pipeline


# Upgrade to a higher-quality multi-label emotions model for richer outputs
sentiment_analysis = pipeline(
  "text-classification",
  framework="pt",
  model="joeddav/distilbert-base-uncased-go-emotions-student",
  top_k=None,
  return_all_scores=True
)

def analyze_sentiment(text):
  results = sentiment_analysis(text)
  if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
    flat = results[0]
  else:
    flat = results
  sentiment_results = {item['label']: item['score'] for item in flat}
  return sentiment_results


def get_bubble_shape(sentiment):
  # Define the mapping of sentiments to bubble shapes
  # 0=normal, 1=jagged, 2=thought, 3=idea, 4=boom, 5=square
  bubble_shape_mapping = {
    "disappointment": 0,
    "sadness": 0,
    "annoyance": 1,
    "neutral": 0,
    "disapproval": 0,
    "realization": 3,  # idea bubble for realizations
    "nervousness": 1,
    "approval": 0,
    "joy": 0,
    "anger": 4,  # boom bubble for anger
    "embarrassment": 0,
    "caring": 0,
    "remorse": 0,
    "disgust": 1,
    "grief": 0,
    "confusion": 2,  # thought cloud for confusion
    "relief": 0,
    "desire": 2,  # thought cloud for desires
    "admiration": 0,
    "optimism": 3,  # idea bubble for optimism
    "fear": 4,  # boom bubble for fear
    "love": 0,
    "excitement": 4,  # boom bubble for excitement
    "curiosity": 2,  # thought cloud for curiosity
    "amusement": 1,
    "surprise": 4,  # boom bubble for surprise
    "gratitude": 0,
    "pride": 0
  }

  shape_names = ["normal", "jagged", "thought", "idea", "boom", "square"]
  shape_index = bubble_shape_mapping.get(sentiment, 0)
  return shape_names[shape_index]


def display_sentiment_results(sentiment_results, option):
  sentiment_text = ""
  for sentiment, score in sentiment_results.items():
    bubble_shape = get_bubble_shape(sentiment)
    if option == "Sentiment Only":
      sentiment_text += f"{bubble_shape}"
    elif option == "Sentiment + Score":
      sentiment_text += f"{bubble_shape}: {score}\n"
  return sentiment_text


def inference(sub, sentiment_option):
  sentiment_results = analyze_sentiment(sub)
  sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)
  return sentiment_output

def get_bubble_type(dialogue):
    # print(dialogue)
    sentiment_option_choices = ["Sentiment Only", "Sentiment + Score"]
    default_sentiment_option = "Sentiment Only"
    sentiment_result = inference(dialogue, default_sentiment_option)
    # print("Sentiment Analysis Results:", sentiment_result)
    return sentiment_result