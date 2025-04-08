# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# # Load the trained model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = { 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8:'I', 9:'J', 10:'K',11:'L',12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U' , 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z', 26:'0', 27:'1', 28:'2', 29:'3', 30:'4', 31:'5', 32:'6', 33:'7', 34:'8', 35:'9'}  # Update this with your labels

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame and detect hands
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         # Iterate through each detected hand
#         for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             if hand_idx >= 2:
#                 break  # Only process the first two hands

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 x_.append(x)
#                 y_.append(y)

#             # Normalize and append the hand landmarks
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         # Pad with zeros if there is only one hand
#         if len(results.multi_hand_landmarks) < 2:
#             padding = [0] * 42  # Padding for one hand
#             data_aux.extend(padding)

#         # Predict with the model
#         prediction = model.predict([np.asarray(data_aux)])

#         predicted_character = labels_dict[int(prediction[0])]

#         # Draw bounding box and predicted label
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)

#     # Show the resulting frame
#     cv2.imshow('frame', frame)

#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont

# # Load the trained model
# model_dict = pickle.load(open('./newmodel.p', 'rb'))
# model = model_dict['model']

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Mediapipe configurations
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Labels for prediction
# labels_dict = {
#     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
#     12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
#     23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7',
#     34: '8', 35: '9',
#     36: 'baby', 37: 'bike', 38: 'camera', 39: 'cell', 40: 'friend', 41: 'gloves', 42: 'good', 43: 'home',
#     44: 'time', 45: 'time-out'
# }

# # Complete translation dictionary
# translation_dict ={
#     'A': {'Hindi': 'ए', 'Gujarati': 'એ'},
#     'B': {'Hindi': 'बी', 'Gujarati': 'બી'},
#     'C': {'Hindi': 'सी', 'Gujarati': 'સી'},
#     'D': {'Hindi': 'डी', 'Gujarati': 'ડી'},
#     'E': {'Hindi': 'ई', 'Gujarati': 'ઈ'},
#     'F': {'Hindi': 'एफ', 'Gujarati': 'એફ'},
#     'G': {'Hindi': 'जी', 'Gujarati': 'જી'},
#     'H': {'Hindi': 'एच', 'Gujarati': 'એચ'},
#     'I': {'Hindi': 'आई', 'Gujarati': 'આઈ'},
#     'J': {'Hindi': 'जे', 'Gujarati': 'જે'},
#     'K': {'Hindi': 'के', 'Gujarati': 'કે'},
#     'L': {'Hindi': 'एल', 'Gujarati': 'એલ'},
#     'M': {'Hindi': 'एम', 'Gujarati': 'એમ'},
#     'N': {'Hindi': 'एन', 'Gujarati': 'એન'},
#     'O': {'Hindi': 'ओ', 'Gujarati': 'ઓ'},
#     'P': {'Hindi': 'पी', 'Gujarati': 'પી'},
#     'Q': {'Hindi': 'क्यू', 'Gujarati': 'ક્યુ'},
#     'R': {'Hindi': 'आर', 'Gujarati': 'આર'},
#     'S': {'Hindi': 'एस', 'Gujarati': 'એસ'},
#     'T': {'Hindi': 'टी', 'Gujarati': 'ટી'},
#     'U': {'Hindi': 'यू', 'Gujarati': 'યુ'},
#     'V': {'Hindi': 'वी', 'Gujarati': 'વી'},
#     'W': {'Hindi': 'डब्ल्यू', 'Gujarati': 'ડબલ્યુ'},
#     'X': {'Hindi': 'एक्स', 'Gujarati': 'એક્સ'},
#     'Y': {'Hindi': 'वाई', 'Gujarati': 'વાય'},
#     'Z': {'Hindi': 'जेड', 'Gujarati': 'ઝેડ'},
#     '0': {'Hindi': '०', 'Gujarati': '૦'},
#     '1': {'Hindi': '१', 'Gujarati': '૧'},
#     '2': {'Hindi': '२', 'Gujarati': '૨'},
#     '3': {'Hindi': '३', 'Gujarati': '૩'},
#     '4': {'Hindi': '४', 'Gujarati': '૪'},
#     '5': {'Hindi': '५', 'Gujarati': '૫'},
#     '6': {'Hindi': '६', 'Gujarati': '૬'},
#     '7': {'Hindi': '७', 'Gujarati': '૭'},
#     '8': {'Hindi': '८', 'Gujarati': '૮'},
#     '9': {'Hindi': '९', 'Gujarati': '૯'},
#     'baby': {'Hindi': 'बच्चा', 'Gujarati': 'બાળક'},
#     'bike': {'Hindi': 'बाइक', 'Gujarati': 'બાઇક'},
#     'camera': {'Hindi': 'कैमरा', 'Gujarati': 'કેમેરા'},
#     'cell': {'Hindi': 'सेल', 'Gujarati': 'સેલ'},
#     'friend': {'Hindi': 'दोस्त', 'Gujarati': 'મિત્ર'},
#     'gloves': {'Hindi': 'दस्ताने', 'Gujarati': 'ગ્લવ્સ'},
#     'good': {'Hindi': 'अच्छा', 'Gujarati': 'સારું'},
#     'home': {'Hindi': 'घर', 'Gujarati': 'ઘર'},
#     'time': {'Hindi': 'समय', 'Gujarati': 'સમય'},
#     'time-out': {'Hindi': 'समय समाप्त', 'Gujarati': 'ટાઇમ-આઉટ'}
# }

# # Load custom fonts
# font_path_hindi = r'C:\Users\Dewansh\Desktop\sign-language-detector-python-master\fonts\NotoSansDevanagari-VariableFont_wdth,wght.ttf'
# font_path_gujarati = r'C:\Users\Dewansh\Desktop\sign-language-detector-python-master\fonts\NotoSansGujarati-VariableFont_wdth,wght.ttf'
# # Updated font sizes
# font_hindi = ImageFont.truetype(font_path_hindi, 40)  # Increase font size
# font_gujarati = ImageFont.truetype(font_path_gujarati, 40)
# font_prediction = ImageFont.truetype("arial.ttf", 40)  # For English predictions

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()
#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame and detect hands
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             if hand_idx >= 2:
#                 break  # Process only two hands

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         # Pad with zeros if only one hand is detected
#         if len(results.multi_hand_landmarks) < 2:
#             padding = [0] * 42
#             data_aux.extend(padding)

#         # Predict the gesture
#         prediction = model.predict([np.asarray(data_aux)])
#         predicted_character = labels_dict[int(prediction[0])]

#         # Get translations
#         hindi_translation = translation_dict.get(predicted_character, {}).get('Hindi', '')
#         gujarati_translation = translation_dict.get(predicted_character, {}).get('Gujarati', '')

#         # Draw bounding box and text
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#         x2 = int(max(x_) * W) + 10
#         y2 = int(max(y_) * H) + 10

#         # Convert frame to PIL Image
#         img_pil = Image.fromarray(frame)

#         # Draw translations using Pillow
#         draw = ImageDraw.Draw(img_pil)
#         draw.rectangle([(x1, y1), (x2, y2)], outline="black", width=4)

#         # Adjust vertical spacing for text
#         text_y = y1 - 80  # Start above the bounding box
#         text_gap = 50     # Gap between each text line

#         # Draw the main prediction
#         draw.text((x1, text_y), f"Prediction: {predicted_character}", font=font_prediction, fill=(0, 0, 255))
#         text_y -= text_gap

#         # Draw Hindi translation
#         draw.text((x1, text_y), f"Hindi: {hindi_translation}", font=font_hindi, fill=(0, 255, 0))
#         text_y -= text_gap

#         # Draw Gujarati translation
#         draw.text((x1, text_y), f"Gujarati: {gujarati_translation}", font=font_gujarati, fill=(255, 0, 0))

#         # Convert back to OpenCV format
#         frame = np.array(img_pil)

#     # Display the frame
#     cv2.imshow('Sign Language Detection', frame)

#     # Exit loop on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load the trained model
model_dict = pickle.load(open('./newmodel10words60_70.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Mediapipe configurations
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels for prediction
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
    23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7',
    34: '8', 35: '9',
    36: 'baby', 37: 'bike', 38: 'camera', 39: 'cell', 40: 'friend', 41: 'gloves', 42: 'good', 43: 'home',
    44: 'time', 45: 'time-out',46:'cash',47:'Clean',48:'Day',49:'Door',50:'Dumbell',51:'Eat',52:'Fan',53:'Fat',54:'Fish',55:'Flower',
    56:'Four(Cricket)',57:'In',58:'Man',59:'Namaste',60:'Nice',61:'Add',62:'Bath',63:'Diwali',64:'Gun',65:'Hear',66:'Hug',67:'Meet',68:'My',69:'On',70:'Spoon'
}

# Complete translation dictionary
translation_dict = {
    'A': {'Hindi': 'ए', 'Gujarati': 'એ'},
    'B': {'Hindi': 'बी', 'Gujarati': 'બી'},
    'C': {'Hindi': 'सी', 'Gujarati': 'સી'},
    'D': {'Hindi': 'डी', 'Gujarati': 'ડી'},
    'E': {'Hindi': 'ई', 'Gujarati': 'ઈ'},
    'F': {'Hindi': 'एफ', 'Gujarati': 'એફ'},
    'G': {'Hindi': 'जी', 'Gujarati': 'જી'},
    'H': {'Hindi': 'एच', 'Gujarati': 'એચ'},
    'I': {'Hindi': 'आई', 'Gujarati': 'આઈ'},
    'J': {'Hindi': 'जे', 'Gujarati': 'જે'},
    'K': {'Hindi': 'के', 'Gujarati': 'કે'},
    'L': {'Hindi': 'एल', 'Gujarati': 'એલ'},
    'M': {'Hindi': 'एम', 'Gujarati': 'એમ'},
    'N': {'Hindi': 'एन', 'Gujarati': 'એન'},
    'O': {'Hindi': 'ओ', 'Gujarati': 'ઓ'},
    'P': {'Hindi': 'पी', 'Gujarati': 'પી'},
    'Q': {'Hindi': 'क्यू', 'Gujarati': 'ક્યુ'},
    'R': {'Hindi': 'आर', 'Gujarati': 'આર'},
    'S': {'Hindi': 'एस', 'Gujarati': 'એસ'},
    'T': {'Hindi': 'टी', 'Gujarati': 'ટી'},
    'U': {'Hindi': 'यू', 'Gujarati': 'યુ'},
    'V': {'Hindi': 'वी', 'Gujarati': 'વી'},
    'W': {'Hindi': 'डब्ल्यू', 'Gujarati': 'ડબલ્યુ'},
    'X': {'Hindi': 'एक्स', 'Gujarati': 'એક્સ'},
    'Y': {'Hindi': 'वाई', 'Gujarati': 'વાય'},
    'Z': {'Hindi': 'जेड', 'Gujarati': 'ઝેડ'},
    '0': {'Hindi': '०', 'Gujarati': '૦'},
    '1': {'Hindi': '१', 'Gujarati': '૧'},
    '2': {'Hindi': '२', 'Gujarati': '૨'},
    '3': {'Hindi': '३', 'Gujarati': '૩'},
    '4': {'Hindi': '४', 'Gujarati': '૪'},
    '5': {'Hindi': '५', 'Gujarati': '૫'},
    '6': {'Hindi': '६', 'Gujarati': '૬'},
    '7': {'Hindi': '७', 'Gujarati': '૭'},
    '8': {'Hindi': '८', 'Gujarati': '૮'},
    '9': {'Hindi': '९', 'Gujarati': '૯'},
    'baby': {'Hindi': 'बच्चा', 'Gujarati': 'બાળક'},
    'bike': {'Hindi': 'बाइक', 'Gujarati': 'બાઇક'},
    'camera': {'Hindi': 'कैमरा', 'Gujarati': 'કેમેરા'},
    'cell': {'Hindi': 'सेल', 'Gujarati': 'સેલ'},
    'friend': {'Hindi': 'दोस्त', 'Gujarati': 'મિત્ર'},
    'gloves': {'Hindi': 'दस्ताने', 'Gujarati': 'ગ્લવ્સ'},
    'good': {'Hindi': 'अच्छा', 'Gujarati': 'સારું'},
    'home': {'Hindi': 'घर', 'Gujarati': 'ઘર'},
    'time': {'Hindi': 'समय', 'Gujarati': 'સમય'},
    'time-out': {'Hindi': 'समय समाप्त', 'Gujarati': 'ટાઇમ-આઉટ'},
    'cash': {'Hindi': 'नकद', 'Gujarati': 'નકદ'},
    'Clean': {'Hindi': 'साफ', 'Gujarati': 'સાફ'},
    'Day': {'Hindi': 'दिन', 'Gujarati': 'દિવસ'},
    'Door': {'Hindi': 'दरवाजा', 'Gujarati': 'દ્વાર'},
    'Dumbell': {'Hindi': 'डम्बल', 'Gujarati': 'ડમ્બેલ'},
    'Eat': {'Hindi': 'खाना', 'Gujarati': 'ખાવું'},
    'Fan': {'Hindi': 'पंखा', 'Gujarati': 'પંખો'},
    'Fat': {'Hindi': 'मोटा', 'Gujarati': 'મોટું'},
    'Fish': {'Hindi': 'मछली', 'Gujarati': 'માછલી'},
    'Flower': {'Hindi': 'फूल', 'Gujarati': 'ફૂલ'},
    'Four(Cricket)': {'Hindi': 'चार (क्रिकेट)', 'Gujarati': 'ચાર (ક્રિકેટ)'},
    'In': {'Hindi': 'में', 'Gujarati': 'માં'},
    'Man': {'Hindi': 'आदमी', 'Gujarati': 'પુરુષ'},
    'Namaste': {'Hindi': 'नमस्ते', 'Gujarati': 'નમસ્તે'},
    'Nice': {'Hindi': 'अच्छा', 'Gujarati': 'સારો'},
    'Add': {'Hindi': 'जोड़ना', 'Gujarati': 'ઉમેરો'},
    'Bath': {'Hindi': 'स्नान', 'Gujarati': 'સ્નાન'},
    'Diwali': {'Hindi': 'दीवाली', 'Gujarati': 'દિવાળી'},
    'Gun': {'Hindi': 'बंदूक', 'Gujarati': 'બંદૂક'},
    'Hear': {'Hindi': 'सुनना', 'Gujarati': 'સાંભળવું'},
    'Hug': {'Hindi': 'गले लगाना', 'Gujarati': 'કોલે'},
    'Meet': {'Hindi': 'मिलना', 'Gujarati': 'મળવું'},
    'My': {'Hindi': 'मेरा', 'Gujarati': 'મારું'},
    'On': {'Hindi': 'पर', 'Gujarati': 'પર'},
    'Spoon': {'Hindi': 'चम्मच', 'Gujarati': 'ચમચી'}
}



# Load custom fonts
font_path_hindi = r'C:\Users\Dewansh\Desktop\sign-language-detector-python-master\fonts\NotoSansDevanagari-VariableFont_wdth,wght.ttf'
font_path_gujarati = r'C:\Users\Dewansh\Desktop\sign-language-detector-python-master\fonts\NotoSansGujarati-VariableFont_wdth,wght.ttf'
font_hindi = ImageFont.truetype(font_path_hindi, 40)  # Increase font size
font_gujarati = ImageFont.truetype(font_path_gujarati, 40)
font_prediction = ImageFont.truetype("arial.ttf", 40)  # For English predictions

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:
                break  # Process only two hands

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Pad with zeros if only one hand is detected
        if len(results.multi_hand_landmarks) < 2:
            padding = [0] * 42
            data_aux.extend(padding)

        # Predict the gesture
        prediction = model.predict([np.asarray(data_aux)])
        predicted_index = int(prediction[0])  # Convert prediction to integer
        predicted_character = labels_dict.get(predicted_index, 'Unknown')  # Map to label

        # Debug: Print the prediction
        print(f"Prediction Index: {predicted_index}, Predicted Character: {predicted_character}")

        # Get translations
        hindi_translation = translation_dict.get(predicted_character, {}).get('Hindi', 'Translation not available')
        gujarati_translation = translation_dict.get(predicted_character, {}).get('Gujarati', 'Translation not available')

        # Draw bounding box and text
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Convert frame to PIL Image
        img_pil = Image.fromarray(frame)

        # Draw translations using Pillow
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([(x1, y1), (x2, y2)], outline="black", width=4)

        # Adjust vertical spacing for text
        text_y = y1 - 80  # Start above the bounding box
        text_gap = 50     # Gap between each text line

        # Draw the main prediction
        draw.text((x1, text_y), f"Prediction: {predicted_character}", font=font_prediction, fill=(0, 0, 255))
        text_y -= text_gap

        # Draw Hindi translation
        draw.text((x1, text_y), f"Hindi: {hindi_translation}", font=font_hindi, fill=(0, 255, 0))
        text_y -= text_gap

        # Draw Gujarati translation
        draw.text((x1, text_y), f"Gujarati: {gujarati_translation}", font=font_gujarati, fill=(255, 0, 0))

        # Convert back to OpenCV format
        frame = np.array(img_pil)

    # Display the frame
    cv2.imshow('Sign Language Detection', frame)

    # Exit loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()