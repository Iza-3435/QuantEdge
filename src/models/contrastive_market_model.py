"""Contrastive Learning for Text-Price Alignment (Market CLIP)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class MarketSample:
    """Paired text-price sample"""
    text: str  # News headline
    price_features: np.ndarray  # Price movement features
    label: float  # Actual return


class TextEncoder(nn.Module):
    """Encode news/text into embeddings"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(384, 256)  # Project to shared space

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        return F.normalize(self.projection(pooled), dim=-1)


class PriceEncoder(nn.Module):
    """Encode price movements into embeddings"""

    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

    def forward(self, price_features):
        return F.normalize(self.encoder(price_features), dim=-1)


class MarketCLIP(nn.Module):
    """
    CLIP-style contrastive learning for markets.

    Learns: "This news text" <-> "This price movement"
    """

    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.price_encoder = PriceEncoder()
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text_inputs, price_features):
        """
        text_inputs: (batch_size, seq_len)
        price_features: (batch_size, feature_dim)
        """
        # Encode both modalities
        text_embeds = self.text_encoder(**text_inputs)
        price_embeds = self.price_encoder(price_features)

        # Compute similarity matrix (like CLIP)
        logits = text_embeds @ price_embeds.T * self.temperature.exp()

        return logits

    def contrastive_loss(self, logits):
        """
        Contrastive loss: matching pairs should have high similarity,
        non-matching pairs should have low similarity.
        """
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        loss_text = F.cross_entropy(logits, labels)
        loss_price = F.cross_entropy(logits.T, labels)

        return (loss_text + loss_price) / 2


class MarketContrastiveTrainer:
    """Train the Market CLIP model"""

    def __init__(self, model: MarketCLIP, lr: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def prepare_batch(self, texts: List[str], price_features: List[np.ndarray]) -> Tuple:
        """Prepare batch for training"""
        # Tokenize text
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        # Convert price features
        price_tensor = torch.FloatTensor(np.array(price_features)).to(self.device)

        return text_inputs, price_tensor

    def train_step(self, texts: List[str], price_features: List[np.ndarray]) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        text_inputs, price_tensor = self.prepare_batch(texts, price_features)

        logits = self.model(text_inputs, price_tensor)
        loss = self.model.contrastive_loss(logits)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, samples: List[MarketSample], epochs: int = 10, batch_size: int = 32):
        """Train the model"""
        print(f"Training on {len(samples)} samples for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0
            batches = 0

            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i+batch_size]

                texts = [s.text for s in batch_samples]
                price_features = [s.price_features for s in batch_samples]

                loss = self.train_step(texts, price_features)
                total_loss += loss
                batches += 1

            avg_loss = total_loss / batches
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def predict_price_movement(self, text: str) -> Dict:
        """
        Given news text, predict likely price movement.

        Returns embedding + can be used to find similar price patterns.
        """
        self.model.eval()

        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        text_embed = self.model.text_encoder(**text_inputs)

        return {
            'embedding': text_embed.cpu().numpy(),
            'text': text
        }

    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Loaded model from {path}")


def create_training_samples(df, news_data: List[Dict]) -> List[MarketSample]:
    """
    Create training samples from price data + news.

    Each sample: (news text, price features before news, price movement after news)
    """
    samples = []

    for news_item in news_data:
        date = news_item['date']
        text = news_item['headline']

        # Find corresponding price data
        if date not in df.index:
            continue

        idx = df.index.get_loc(date)
        if idx < 20 or idx >= len(df) - 5:
            continue

        # Features before news
        window = df.iloc[idx-20:idx]
        returns = window['Close'].pct_change().values
        price_features = np.array([
            returns.mean(),
            returns.std(),
            returns[-1],
            window['RSI'].iloc[-1] / 100 if 'RSI' in window else 0.5,
            window['Volume'].iloc[-1] / window['Volume'].mean()
        ])

        # Outcome after news
        future_return = df['Close'].iloc[idx+5] / df['Close'].iloc[idx] - 1

        samples.append(MarketSample(
            text=text,
            price_features=price_features,
            label=future_return
        ))

    return samples


if __name__ == "__main__":
    print("Testing Market CLIP (Contrastive Learning)...")
    print("="*70)

    # Create dummy training samples
    print("\n1. Creating training samples...")

    sample_news = [
        "Apple stock surges on strong earnings beat",
        "Tech sector faces regulatory pressure",
        "Market rallies on Fed dovish stance",
        "Economic data shows slowdown in growth",
        "Apple announces major product launch"
    ]

    samples = []
    for i, text in enumerate(sample_news):
        # Dummy price features
        price_features = np.random.randn(5)
        samples.append(MarketSample(
            text=text,
            price_features=price_features,
            label=np.random.uniform(-0.05, 0.05)
        ))

    print(f"Created {len(samples)} samples")

    # Initialize model
    print("\n2. Initializing Market CLIP model...")
    model = MarketCLIP()
    trainer = MarketContrastiveTrainer(model, lr=1e-4)

    # Train
    print("\n3. Training...")
    trainer.train(samples, epochs=5, batch_size=2)

    # Test prediction
    print("\n4. Testing prediction...")
    test_text = "Breaking: Apple announces record revenue"
    prediction = trainer.predict_price_movement(test_text)

    print(f"\nText: {test_text}")
    print(f"Embedding shape: {prediction['embedding'].shape}")
    print(f"✅ Can now match text to historical price patterns")

    print("\n" + "="*70)
    print("HOW THIS WORKS:")
    print("="*70)
    print("""
1. Model learns: "This type of news" → "This type of price movement"
2. New news arrives → Model embeds it
3. Find similar historical patterns with similar embeddings
4. Predict outcome based on what happened historically

Benefits:
  • Zero-shot: Works on news it hasn't seen
  • Multimodal: Combines text + price data
  • Interpretable: Can show similar historical examples
    """)
    print("="*70)
