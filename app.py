from flask import Flask, request, jsonify
from source.handle_image import get_uri, read_image, ndarrayToPIL
from source.image_enhancement_mamdani import FuzzyContrastEnhance
from source.image_enhancement_sugeno import FuzzySugenoContrastEnhance
from source.image_enhancement_tsukamoto import FuzzyTsukamotoContrastEnhance
from source.image_enhancement import combineMamdani, combineSugeno, combineTsukamoto, combineHistogram, histogramEqualization, clahe, calculate, calculateCombine
from flask_cors import CORS

app = Flask(__name__)
# CORS(app, origins="https://fuzzy-image-enhancement.vercel.app")
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'This is main page'})
    

@app.route("/api/process_image", methods=["POST"])
def process():
    if request.method == "POST":
        file = request.files["file-image"]

        if file:
            # Enhancement
            histogram_enhancement = get_uri(ndarrayToPIL(histogramEqualization(read_image(file))))
            mamdani_enhancement = get_uri(ndarrayToPIL(FuzzyContrastEnhance(read_image(file))))
            sugeno_enhancement = get_uri(ndarrayToPIL(FuzzySugenoContrastEnhance(read_image(file))))
            tsukamoto_enhancement = get_uri(ndarrayToPIL(FuzzyTsukamotoContrastEnhance(read_image(file))))

            # Clahe
            clahe_enhancement = get_uri(ndarrayToPIL(clahe(read_image(file))))

            # Combine Enhancement
            enhanced_image = get_uri(ndarrayToPIL(combineMamdani(read_image(file))))
            sugeno_enhanced_image = get_uri(ndarrayToPIL(combineSugeno(read_image(file))))
            tsukamoto_enhanced_image = get_uri(ndarrayToPIL(combineTsukamoto(read_image(file))))
            histogram_enhanced_image = get_uri(ndarrayToPIL(combineHistogram(read_image(file))))

            # PNSR Enhacement
            clahe_pnsr, histogram_pnsr, mamdani_pnsr, sugeno_pnsr, tsukamoto_pnsr = calculate(read_image(file))

            # PNSR Combine
            combine_histogram_pnsr, combine_mamdani_pnsr, combine_sugeno_pnsr, combine_tsukamoto_pnsr = calculateCombine(read_image(file))

            response = jsonify({
                'message': 'Image processed successfully',
                "mamdani_enhancement": mamdani_enhancement, 
                "sugeno_enhancement": sugeno_enhancement, 
                "tsukamoto_enhancement": tsukamoto_enhancement, 
                "histogram_enhancement": histogram_enhancement,
                "clahe_enhancement": clahe_enhancement,
                "encoded_image": enhanced_image, 
                "sugeno_encoded_image": sugeno_enhanced_image, 
                "tsukamoto_encoded_image": tsukamoto_enhanced_image, 
                "histogram_encoded_image": histogram_enhanced_image,
                "clahe_pnsr": clahe_pnsr,
                "histogram_pnsr": histogram_pnsr,
                "mamdani_pnsr": mamdani_pnsr,
                "sugeno_pnsr": sugeno_pnsr,
                "tsukamoto_pnsr": tsukamoto_pnsr,
                "combine_histogram_pnsr": combine_histogram_pnsr,
                "combine_mamdani_pnsr": combine_mamdani_pnsr,
                "combine_sugeno_pnsr": combine_sugeno_pnsr,
                "combine_tsukamoto_pnsr": combine_tsukamoto_pnsr,
            })

            # origin = request.headers.get('Origin')
            # if origin in ['https://fuzzy-image-enhancement.vercel.app/', 'https://fuzzy-image-enhancement-marwahns-projects.vercel.app/', 'https://fuzzy-image-enhancement-git-main-marwahns-projects.vercel.app/', 'http://127.0.0.1:5500']:
            #     response.headers.add('Access-Control-Allow-Origin', origin)
            #     return response

            return response
            
    # Handle other cases or return an error message
    return jsonify({'message': 'Invalid request'})


# Handle 404 page
@app.errorhandler(404)
def page_not_found(error):
    return jsonify({'message': 'Page Not Found'}), 404


if __name__ == "__main__":
    app.run(debug=True)