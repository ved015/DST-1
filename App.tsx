import React, { useEffect, useState } from "react";
import { View, Text, Button, Image, ActivityIndicator, Alert } from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as FileSystem from "expo-file-system";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native"; // Initializes the native TensorFlow backend
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import * as jpeg from "jpeg-js";

export default function App() {
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  // Request permission to access the photo library.
  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== "granted") {
        Alert.alert("Permission required", "We need permission to access your photo library.");
        setHasPermission(false);
      } else {
        setHasPermission(true);
      }
    })();
  }, []);

  // Load TensorFlow and the model on mount.
  useEffect(() => {
    async function loadModel() {
      try {
        console.log("Initializing TensorFlow...");
        await tf.ready();
        console.log("TensorFlow Ready!");

        // Load model files from the assets folder.
        const modelJson = require("./assets/mnist_tfjs2/model.json");
        const modelWeights = require("./assets/mnist_tfjs2/group1-shard1of1.bin");

        // Load the model as a GraphModel.
        const loadedModel = await tf.loadGraphModel(bundleResourceIO(modelJson, [modelWeights]));
        console.log("Model Loaded Successfully!");
        setModel(loadedModel);
      } catch (error) {
        console.error("Error loading model:", error);
      } finally {
        setLoading(false);
      }
    }
    loadModel();
  }, []);

  // Function to pick an image.
  const pickImage = async () => {
    if (!hasPermission) {
      Alert.alert("Permission Denied", "You need to allow access to your photo library.");
      return;
    }
    try {
      console.log("Launching image library...");
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 1,
      });
      console.log("ImagePicker result:", result);

      // Handle both newer and older expo-image-picker responses.
      let uri: string | undefined = undefined;
      if (result.canceled !== undefined) {
        if (!result.canceled && result.assets && result.assets.length > 0) {
          uri = result.assets[0].uri;
        }
      } else if (result.cancelled !== undefined) {
        if (!result.cancelled && result.uri) {
          uri = result.uri;
        }
      }
      if (uri) {
        console.log("Image URI selected:", uri);
        setImage(uri);
        predictImage(uri);
      } else {
        console.log("No image selected.");
      }
    } catch (error) {
      console.error("Error picking image:", error);
    }
  };

  // Function to preprocess the image and run prediction.
  const predictImage = async (uri: string) => {
    if (!model) return;
    try {
      console.log("Starting prediction for image:", uri);
      
      // Read and decode image
      const imgB64 = await FileSystem.readAsStringAsync(uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const imgBuffer = Buffer.from(imgB64, "base64");
      const rawImageData = jpeg.decode(imgBuffer, { useTArray: true });
  
      // Create tensor from image data
      let imgTensor = tf.browser.fromPixels(rawImageData).toFloat();
      console.log("Original tensor shape:", imgTensor.shape);
  
      // Convert to grayscale and maintain channel dimension
      let grayTensor = tf.image.rgbToGrayscale(imgTensor);
      console.log("Grayscale tensor shape:", grayTensor.shape);
  
      // Calculate aspect ratio preserving resize
      const [height, width] = grayTensor.shape.slice(0, 2);
      const maxDim = Math.max(height, width);
      const scaleFactor = 20 / maxDim;
      const newHeight = Math.floor(height * scaleFactor);
      const newWidth = Math.floor(width * scaleFactor);
  
      // Resize with bilinear interpolation
      let resizedTensor = tf.image.resizeBilinear(grayTensor, [newHeight, newWidth]);
      console.log("Resized tensor shape:", resizedTensor.shape);
  
      // Calculate mean on the digit region before padding
      const regionMean = resizedTensor.mean().dataSync()[0];
      console.log("Resized region mean:", regionMean);
  
      // Pad to 28x28 with black borders
      const padVert = 28 - newHeight;
      const padHorz = 28 - newWidth;
      const paddedTensor = tf.pad(resizedTensor, [
        [Math.floor(padVert/2), Math.ceil(padVert/2)], // Vertical
        [Math.floor(padHorz/2), Math.ceil(padHorz/2)], // Horizontal
        [0, 0] // Channels
      ]);
      console.log("Padded tensor shape:", paddedTensor.shape);
  
      // Normalize and invert if needed
      let processedTensor = paddedTensor.div(255.0);
      if (regionMean > 0.3) { // More sensitive inversion threshold
        processedTensor = tf.sub(1, processedTensor);
        console.log("Image inverted");
      }
  
      // Visualize the processed tensor
      // const visualizationTensor = processedTensor.mul(255).toInt();
      // tf.browser.toPixels(visualizationTensor.squeeze() as tf.Tensor2D, document.createElement('canvas'))
      //   .then(data => {
      //     console.log("Processed image preview available");
      //     // You can implement actual image preview here if needed
      //   });
  
      // Flatten and format for dense model
      const inputTensor = processedTensor.reshape([1, 784]);
      console.log("Input tensor values (sample):", 
        Array.from(inputTensor.dataSync()).slice(0, 10).map(v => v.toFixed(2)));
  
      // Run prediction
      const prediction = model.execute(inputTensor) as tf.Tensor;
      const scores = Array.from(prediction.dataSync()).map(v => Number(v.toFixed(3)));
      const predictedClass = scores.indexOf(Math.max(...scores));
      
      console.log("Prediction scores:", scores);
      setPrediction(predictedClass);
  
    } catch (error) {
      console.error("Error processing image:", error);
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
      {loading ? (
        <ActivityIndicator size="large" />
      ) : (
        <Text style={{ marginBottom: 20 }}>Model Loaded</Text>
      )}
      {image && (
        <Image
          source={{ uri: image }}
          style={{ width: 100, height: 100, marginBottom: 20 }}
        />
      )}
      <Button title="Pick an image" onPress={pickImage} />
      {prediction !== null && (
        <Text style={{ marginTop: 20, fontSize: 18 }}>Prediction: {prediction}</Text>
      )}
    </View>
  );
}
