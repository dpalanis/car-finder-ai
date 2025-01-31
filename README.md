# car-finder-ai
Utilized the most advanced technology to build an API which is capable of recognizing the Make, Model with great accuracy.

Image Labeling Tool: Roboflow
Base Model: Yolo11
Required Folders: 
  FolderName: <parent_folder>/output
  FolderName: <parent_folder>/static -- input images.

This repository contains a Flask-based web application for object detection using the YOLO (You Only Look Once) model. The application can process images, detecting objects and displaying the results.

Steps:

1. Deploy the car-finder-ai application.
2. Clone the repository.
3. git clone URL
4. cd cloned directory path
5. Download the YOLO weights from this link and place them in the object_detection directory.
6. Annotate your images using a labeling tool like Roboflow.
7. Train the model (use Google Colab).
8. Run the application using python app.py (the application will be available at http://localhost:8000).
9. Deploy the car-finder-react-ui.
10. Modify the server endpoint in the code and input the image path.
11. Run npm start.
12. Upload an image and verify the results.
  
