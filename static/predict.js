document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const data = {
        commodity_name: document.getElementById('commodity').value,
        state: document.getElementById('state').value,
        district: document.getElementById('district').value,
        market: document.getElementById('market').value,
        date: document.getElementById('date').value
    };

    const response = await fetch('/predict_price', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });

    const result = await response.json();
    document.getElementById('result').textContent = 'Prezzo Modal Predetto: ' + result.predicted_modal_price.toFixed(2);
});