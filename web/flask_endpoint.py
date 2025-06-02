from flask import Flask, request, jsonify
from knn.knn import get_final_score, collect_ngram_times

app = Flask(__name__)
data_dir = "data"
all_patterns = collect_ngram_times(data_dir)

@app.route('/receive-json', methods=['POST'])
def receive_json():
    data = request.get_json()
    ngram_times = data["ngram_times"]
    final_score = get_final_score(all_patterns, ngram_times)
    print("Received JSON:", data)
    return jsonify({"status": "received", "score": final_score}), 200

if __name__ == '__main__':
    app.run(debug=True)
