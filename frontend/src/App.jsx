const API_URL = "http://127.0.0.1:8000";
import { useState } from "react";
import "./App.css";

const initialOrder = {
  product_category: "Electronics",
  price_inr: 16689,
  discount_pct: 6.7,
  payment_method: "COD",
  customer_tenure_days: 309,
  num_previous_orders: 11,
  num_previous_returns: 3,
  delivery_distance_km: 166.4,
  delivery_days: 9,
  is_weekend_order: 0,
  rating_given: 2,
};

function App() {
  const [agentQuery, setAgentQuery] = useState("");
  const [agentAnswer, setAgentAnswer] = useState("");
  const [agentLoading, setAgentLoading] = useState(false);
  const [showOrderForm, setShowOrderForm] = useState(false);
  const [order, setOrder] = useState(initialOrder);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showImagePanel, setShowImagePanel] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState("");   
  const handleChange = (event) => {
  const { name, value } = event.target;

    setOrder((current) => ({
      ...current,
      [name]:
        name === "product_category" || name === "payment_method"
          ? value
          : Number(value),
    }));
  };

  const analyzeOrder = async (event) => {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict-return",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(order),
        }
      );

      if (!response.ok) {
        throw new Error("API request failed.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        "Could not connect to the AI backend. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleImageSelect = (event) => {
  const file = event.target.files?.[0];

  if (!file) return;

  setSelectedImage(file);
  setImagePreview(URL.createObjectURL(file));
  setImageResult(null);
  setImageError("");
};

const classifyImage = async () => {
  if (!selectedImage) {
    setImageError("Please select an image first.");
    return;
  }

  setImageLoading(true);
  setImageError("");
  setImageResult(null);

  try {
    const formData = new FormData();
    formData.append("file", selectedImage);

    const response = await fetch(
      "http://127.0.0.1:8000/predict-image",
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      throw new Error("Image prediction failed.");
    }

    const data = await response.json();

    setImageResult(data);
  } catch (err) {
    setImageError(
      "Could not connect to the AI backend. Make sure FastAPI is running."
    );
  } finally {
    setImageLoading(false);
  }
};

async function askAgent() {
  if (!agentQuery.trim()) return;

  setAgentLoading(true);
  setAgentAnswer("");

  try {
    const response = await fetch(
      `${API_URL}/agent/chat`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: agentQuery,
          order_features: order,
       }),
      }
    );

    if (!response.ok) {
      throw new Error("Agent request failed");
    }

    const data = await response.json();

    setAgentAnswer(data.answer);
  } catch (error) {
    console.error(error);
    setAgentAnswer(
      "Could not connect to the AI agent."
    );
  } finally {
    setAgentLoading(false);
  }
}

  return (
    <div className="app">
      <header className="navbar">
        <div className="brand">
          <div className="brand-mark">FI</div>

          <div>
            <h1>Flipkart Intelligence</h1>
            <span>ML Decision Platform</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          API Online
        </div>
      </header>

      <main className="dashboard">
        <section className="hero">
          <p className="eyebrow">AI-POWERED ORDER ANALYTICS</p>

          <h2>
            Turn order data into
            <span> intelligent decisions.</span>
          </h2>

          <p className="hero-text">
            Predict return risk and classify products using
            machine-learning models through 
            one unified dashboard.
          </p>
        </section>
<section className="agent-panel">
  <div className="panel-header">
    <div>
      <span className="card-label">
        AI SUPPORT AGENT
      </span>

      <h3>Ask Flipkart Intelligence</h3>
    </div>

    <span className="agent-status">
      ● Online
    </span>
  </div>

  <p className="agent-description">
    Ask about return policies, return risk,
    or product classification.
  </p>

  <div className="agent-input-row">
    <input
      type="text"
      placeholder="Ask something..."
      value={agentQuery}
      onChange={(e) =>
        setAgentQuery(e.target.value)
      }
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          askAgent();
        }
      }}
    />

    <button
      onClick={askAgent}
      disabled={agentLoading}
    >
      {agentLoading
        ? "Thinking..."
        : "Ask Agent →"}
    </button>
  </div>

  {agentAnswer && (
    <div className="agent-response">
      <span>AGENT RESPONSE</span>

      <p>{agentAnswer}</p>
    </div>
  )}
</section>

        <section className="cards">
          <div className="card">
            <div className="card-header">
              <div>
                <span className="card-label">MODEL 01</span>
                <h3>Return Risk</h3>
              </div>

              <div className="icon">↗</div>
            </div>

            <p>
              Analyze an order and estimate the probability that
              it will be returned.
            </p>

            <button
              onClick={() => setShowOrderForm(true)}
            >
              Analyze Order
              <span>→</span>
            </button>
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <span className="card-label">MODEL 02</span>
                <h3>Product Vision</h3>
              </div>

              <div className="icon">✦</div>
            </div>

            <p>
              Upload a product image and let the CNN identify
              the most likely product category.
            </p>

            <button onClick={() => setShowImagePanel(true)}>
                   Classify Image
             <span>→</span>
            </button>
          </div>
        </section>

        {showOrderForm && (
          <section className="order-panel">
            <div className="panel-header">
              <div>
                <span className="card-label">MODEL 01</span>
                <h3>Return Risk Analysis</h3>
              </div>

              <button
                className="close-button"
                onClick={() => setShowOrderForm(false)}
              >
                ×
              </button>
            </div>

            <form onSubmit={analyzeOrder}>
              <div className="form-grid">
                <label>
                  Product category
                  <select
                    name="product_category"
                    value={order.product_category}
                    onChange={handleChange}
                  >
                    <option>Electronics</option>
                    <option>Footwear</option>
                    <option>Apparel</option>
                    <option>Home</option>
                    <option>Beauty</option>
                  </select>
                </label>

                <label>
                  Payment method
                  <select
                    name="payment_method"
                    value={order.payment_method}
                    onChange={handleChange}
                  >
                    <option>COD</option>
                    <option>Prepaid_Card</option>
                    <option>Prepaid_UPI</option>
                    <option>Wallet</option>
                  </select>
                </label>

                <label>
                  Price (₹)
                  <input
                    type="number"
                    name="price_inr"
                    value={order.price_inr}
                    onChange={handleChange}
                  />
                </label>

                <label>
                  Discount (%)
                  <input
                    type="number"
                    step="0.1"
                    name="discount_pct"
                    value={order.discount_pct}
                    onChange={handleChange}
                  />
                </label>

                <label>
                  Customer tenure (days)
                  <input
                    type="number"
                    name="customer_tenure_days"
                    value={order.customer_tenure_days}
                    onChange={handleChange}
                  />
                </label>

                <label>
                  Previous orders
                  <input
                    type="number"
                    name="num_previous_orders"
                    value={order.num_previous_orders}
                    onChange={handleChange}
                  />
                </label>

                <label>
                  Previous returns
                  <input
                    type="number"
                    name="num_previous_returns"
                    value={order.num_previous_returns}
                    onChange={handleChange}
                  />
                </label>

                <label>
                  Delivery distance (km)
                  <input
                    type="number"
                    step="0.1"
                    name="delivery_distance_km"
                    value={order.delivery_distance_km}
                    onChange={handleChange}
                  />
                </label>

                <label>
                  Delivery days
                  <input
                    type="number"
                    name="delivery_days"
                    value={order.delivery_days}
                    onChange={handleChange}
                  />
                </label>

                <label>
                  Weekend order
                  <select
                    name="is_weekend_order"
                    value={order.is_weekend_order}
                    onChange={handleChange}
                  >
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </label>

                <label>
                  Rating given
                  <input
                    type="number"
                    step="0.1"
                    min="1"
                    max="5"
                    name="rating_given"
                    value={order.rating_given}
                    onChange={handleChange}
                  />
                </label>
              </div>

              <button
                className="analyze-button"
                type="submit"
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Run AI Analysis →"}
              </button>
            </form>

            {error && (
              <div className="error-box">
                {error}
              </div>
            )}

            {result && (
              <div className="result-box">
                <span className="card-label">
                  AI PREDICTION
                </span>

                <div className="risk-score">
                  {result.return_probability_percent}%
                </div>

                <h4>{result.prediction}</h4>

                <p>
                  Estimated probability of this order being
                  returned.
                </p>
              </div>
            )}
          </section>
        )}

     
    {/* ALL YOUR RETURN RISK FORM CODE HERE */}

{showImagePanel && (
  <section className="image-panel">
    <div className="panel-header">
      <div>
        <span className="card-label">MODEL 02</span>
        <h3>Product Vision</h3>
      </div>

      <button
        className="close-button"
        onClick={() => setShowImagePanel(false)}
      >
        ×
      </button>
    </div>

    <div className="image-upload-area">
      <input
        id="product-image"
        type="file"
        accept="image/png,image/jpeg,image/jpg"
        onChange={handleImageSelect}
      />

      <label htmlFor="product-image" className="upload-button">
        Choose Product Image
      </label>

      {imagePreview && (
        <div className="preview-container">
          <img
            src={imagePreview}
            alt="Selected product"
            className="image-preview"
          />
        </div>
      )}
    </div>

    <button
      className="analyze-button"
      onClick={classifyImage}
      disabled={!selectedImage || imageLoading}
    >
      {imageLoading
        ? "Classifying..."
        : "Run Vision Analysis →"}
    </button>

    {imageError && (
      <div className="error-box">
        {imageError}
      </div>
    )}

    {imageResult && (
      <div className="result-box">
        <span className="card-label">
          CNN PREDICTION
        </span>

        <div className="vision-result">
          <div>
            <span className="result-label">
              PREDICTED PRODUCT
            </span>

            <div className="prediction-name">
              {imageResult.predicted_class}
            </div>
          </div>

          <div>
            <span className="result-label">
              CONFIDENCE
            </span>

            <div className="risk-score">
              {imageResult.confidence_percent}%
            </div>
          </div>
        </div>
      </div>
    )}
  </section>
)}

        <section className="system">
          <div>
            <span className="card-label">SYSTEM STATUS</span>
            <h3>Models ready for inference</h3>
          </div>

          <div className="model-status">
            <div>
              <span className="status-dot"></span>
              Random Forest
            </div>

            <div>
              <span className="status-dot"></span>
              Fashion CNN
            </div>

            <div>
              <span className="status-dot"></span>
              FastAPI
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}


export default App;