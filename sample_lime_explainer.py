from lime.lime_text import LimeTextExplainer
import numpy as np
import torchtext

# Define class names for binary classification
class_names = ['Negative', 'Positive']

# Load the trained model
model.load_state_dict(torch.load('text_classifier.pt'))
model.eval()

# Tokenizer and LIME explainer
tokenizer = lambda x: [tok.text for tok in torchtext.data.utils.get_tokenizer('spacy')(x)]
explainer = LimeTextExplainer(class_names=class_names)

# Predict function for LIME
def predict_fn(texts):
    tokens = [TEXT.process(tokenizer(text)) for text in texts]
    lengths = torch.tensor([len(t) for t in tokens])
    batch_text = nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    with torch.no_grad():
        preds = torch.sigmoid(model(batch_text, lengths)).numpy()
    return np.stack([1 - preds, preds], axis=1)

# Example input text
sample_text = "The movie was great but the ending was disappointing."

# Explain the prediction
explanation = explainer.explain_instance(
    sample_text, predict_fn, num_features=6
)

# Show the explanation
print("Explanation for the prediction:")
print(explanation.as_list())
explanation.show_in_notebook()

