from flask import Flask, jsonify

# Flask server uses port 5000. To run on website use localhost:5000/api/data
app = Flask(__name__)

@app.route('/api/data', methods=['GET'])

def get_data():
    data = {"message": "Hello from Python backend!"}
    return jsonify(data)

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000, debug=True);