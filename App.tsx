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

  // Request permission to access the media library on mount.
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
        const modelJson = require("./assets/mnist_tfjs/model.json");
        const modelWeights = require("./assets/mnist_tfjs/group1-shard1of1.bin");

        // Load the model as a GraphModel.
        const loadedModel = await tf.loadGraphModel(
          bundleResourceIO(modelJson, [modelWeights])
        );
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
        mediaTypes: ImagePicker.MediaTypeOptions.Images, // Correct usage for images
        allowsEditing: true,
        aspect: [1, 1],
        quality: 1,
      });
      console.log("ImagePicker result:", result);

      // Handle both new and older versions of expo-image-picker's result.
      let uri: string | undefined = undefined;
      if (result.canceled !== undefined) {
        // Newer API: result.canceled (boolean) and result.assets (array)
        if (!result.canceled && result.assets && result.assets.length > 0) {
          uri = result.assets[0].uri;
        }
      } else if (result.cancelled !== undefined) {
        // Older API: result.cancelled (boolean) and result.uri (string)
        if (!result.cancelled && result.uri) {
          uri = result.uri;
        }
      }

      if (uri) {
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
      // Read the image as a base64 string.
      const imgB64 = await FileSystem.readAsStringAsync(uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const imgBuffer = Buffer.from(imgB64, "base64");
      const rawImageData = jpeg.decode(imgBuffer, true);

      // Create a tensor from the image data.
      let imgTensor = tf.browser.fromPixels(rawImageData).toFloat();
      // Resize the image to 28x28.
      imgTensor = tf.image.resizeBilinear(imgTensor, [28, 28]);

      // Convert to grayscale using weighted conversion:
      // gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
      const [red, green, blue] = tf.split(imgTensor, 3, 2);
      const grayTensor = red.mul(0.2989)
        .add(green.mul(0.5870))
        .add(blue.mul(0.1140));

      // Remove the channel dimension: shape becomes [28,28].
      let processedTensor = grayTensor.squeeze([2]);
      // Add a batch dimension and normalize to [0,1]: shape becomes [1,28,28].
      processedTensor = processedTensor.expandDims(0).div(255.0);

      // (Optional) Invert colors if the image's overall brightness is high.
      const meanValue = processedTensor.mean().dataSync()[0];
      console.log("Mean pixel value:", meanValue);
      // If the mean is high (e.g. > 0.5), assume the image is inverted relative to MNIST.
      if (meanValue > 0.5) {
        console.log("Inverting colors...");
        processedTensor = tf.sub(1, processedTensor);
      }

      // Run the model.
      const predictionTensor = model.execute(processedTensor) as tf.Tensor;
      const predictedClass = predictionTensor.argMax(-1).dataSync()[0];
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
