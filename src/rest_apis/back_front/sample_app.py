from flask import Flask,jsonify, request
from products import products

app = Flask(__name__)

@app.route('/ping')
def ping():
    return jsonify({"message":"Pong"})

@app.route('/products')
def getProducts():
    return jsonify({"products":products, "message": "Product's List"})

@app.route('/products/<string:product_name>')
def getProduct(product_name):
    products_found = [product for product in products if product['name'] == product_name ]
    if (len(products_found) > 0):
        return jsonify({"product":products_found[0]})
    return jsonify({"message":"product not found"}) 

@app.route('/products', methods=['POST'])
def createProduct():
    new_product = {
        "name": request.json["name"],
        "price": request.json["price"],
        "quantity": request.json["quantity"]
    }
    products.append(new_product)
    return jsonify({"message":"Product added", "products":products})


if __name__ == '__main__':
    app.run(debug=True, port=4000)