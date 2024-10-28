import json
from ocr_neural_network import OCRNeuralNetwork  # Assuming 'ocr_neural_network' is the correct module

data_matrix = ...  # Define or load your data matrix
data_labels = ...  # Define or load your data labels
train_indices = ...  # Define or load your training indices
test_indices = ...  # Define or load your test indices

# Try various number of hidden nodes and see what performs best
for i in range(5, 50, 5):
    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
    performance = str(test(data_matrix, data_labels, test_indices, nn))
    print("{i} Hidden Nodes: {val}".format(i=i, val=performance))

def test(data_matrix, data_labels, test_indices, nn):
    avg_sum = 0
    for j in range(100):
        correct_guess_count = 0
        for i in test_indices:
            test = data_matrix[i]
            prediction = nn.predict(test)
            if data_labels[i] == prediction:
                correct_guess_count += 1

        avg_sum += (correct_guess_count / float(len(test_indices)))
    return avg_sum / 100

def do_POST(s):
    response_code = 200
    response = ""
    var_len = int(s.headers.get('Content-Length'))
    content = s.rfile.read(var_len)
    payload = json.loads(content)

    if payload.get('train'):
        nn.train(payload['trainArray'])
        nn.save()
    elif payload.get('predict'):
        try:
            response = {
                "type": "test",
                "result": nn.predict(str(payload['image']))
            }
        except:
            response_code = 500
    else:
        response_code = 400

    s.send_response(response_code)
    s.send_header("Content-type", "application/json")
    s.send_header("Access-Control-Allow-Origin", "*")
    s.end_headers()
    if response:
        s.wfile.write(json.dumps(response).encode('utf-8'))
    return