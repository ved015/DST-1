import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  ActivityIndicator,
  Alert,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as FileSystem from "expo-file-system";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native"; // Initializes the native TensorFlow backend
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import * as jpeg from "jpeg-js";

const { width: screenWidth } = Dimensions.get("window");

export default function App() {
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  // gridMatrix: 2D array (28x28) representing the binarized digit
  const [gridMatrix, setGridMatrix] = useState<string[][]>([]);
  const [inputShape, setInputShape] = useState<string>("");
  // New state: probabilities for each digit (0-9)
  const [probabilities, setProbabilities] = useState<number[]>([]);

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
        await tf.ready();
        const modelJson = require("./assets/mnist_tfjs2/model.json");
        const modelWeights = require("./assets/mnist_tfjs2/group1-shard1of1.bin");
        const loadedModel = await tf.loadGraphModel(bundleResourceIO(modelJson, [modelWeights]));
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
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 1,
      });

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
        setImage(uri);
        predictImage(uri);
      }
    } catch (error) {
      console.error("Error picking image:", error);
    }
  };

  // Function to preprocess the image and run prediction.
  const predictImage = async (uri: string) => {
    if (!model) return;
    try {
      const imgB64 = await FileSystem.readAsStringAsync(uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const imgBuffer = Buffer.from(imgB64, "base64");
      const rawImageData = jpeg.decode(imgBuffer, { useTArray: true });

      let imgTensor = tf.browser.fromPixels(rawImageData).toFloat();
      let grayTensor = tf.image.rgbToGrayscale(imgTensor);
      const [height, width] = grayTensor.shape.slice(0, 2);
      setInputShape(`${height} x ${width}`);
      const maxDim = Math.max(height, width);
      const scaleFactor = 28 / maxDim;
      const newHeight = Math.floor(height * scaleFactor);
      const newWidth = Math.floor(width * scaleFactor);
      let resizedTensor = tf.image.resizeBilinear(grayTensor, [newHeight, newWidth]);
      let normalizedTensor = resizedTensor.div(255.0);
      const regionMean = normalizedTensor.mean().dataSync()[0];
      if (regionMean > 0.5) {
        normalizedTensor = tf.sub(1, normalizedTensor);
      }
      const padVert = 28 - newHeight;
      const padHorz = 28 - newWidth;
      const paddedTensor = tf.pad(normalizedTensor, [
        [Math.floor(padVert / 2), Math.ceil(padVert / 2)],
        [Math.floor(padHorz / 2), Math.ceil(padHorz / 2)],
        [0, 0],
      ]);
      const paddedData = paddedTensor.dataSync();
      const arr = Array.from(paddedData);
      arr.sort((a, b) => a - b);
      const medianValue = arr[Math.floor(arr.length / 2)];
      const binaryTensor = paddedTensor.greater(tf.scalar(medianValue)).cast("float32");
      const inputTensor = binaryTensor.reshape([1, 784]);

      // Build the 28x28 grid as a 2D array.
      const data = inputTensor.dataSync();
      let grid: string[][] = [];
      for (let i = 0; i < 28; i++) {
        let row: string[] = [];
        for (let j = 0; j < 28; j++) {
          row.push(data[i * 28 + j].toFixed(0));
        }
        grid.push(row);
      }
      setGridMatrix(grid);

      const predictionTensor = model.execute(inputTensor) as tf.Tensor;
      const scores = Array.from(predictionTensor.dataSync()).map((v) => Number(v.toFixed(2)));
      const predictedClass = scores.indexOf(Math.max(...scores));
      setPrediction(predictedClass);
      // Set probabilities so they can be displayed in a single row.
      setProbabilities(scores);

      tf.dispose([
        imgTensor,
        grayTensor,
        resizedTensor,
        normalizedTensor,
        paddedTensor,
        binaryTensor,
        inputTensor,
      ]);
    } catch (error) {
      console.error("Error processing image:", error);
    }
  };

  return (
    <View style={styles.container}>
      {/* Top 30%: Header */}
      <View style={styles.headerContainer}>
        <Text style={styles.title}>Digit Classifier Model</Text>
        {loading ? (
          <ActivityIndicator size="large" color="#fff" />
        ) : (
          <Text style={styles.modelStatus}>Model Loaded</Text>
        )}
        {inputShape !== "" && (
          <Text style={styles.inputShape}>Input Tensor Shape: {inputShape}</Text>
        )}
        {/* Display probabilities array in a single row */}
        {probabilities.length > 0 && (
          <View style={styles.probabilitiesContainer}>
            <Text style={styles.probabilityRow}>
              {`[${probabilities.join(", ")}]`}
            </Text>
          </View>
        )}
      </View>

      {/* Middle 50%: Grid */}
      <View style={styles.gridContainer}>
        {gridMatrix.length > 0 ? (
          <View style={styles.gridBlock}>
            {gridMatrix.map((row, rowIndex) => (
              <View key={rowIndex} style={styles.gridRow}>
                {row.map((cell, colIndex) => (
                  <View key={colIndex} style={styles.gridCell}>
                    <Text
                      style={[
                        styles.gridCellText,
                        cell === "1" && styles.gridCellTextOne, // Extra styling for cells with value "1"
                      ]}
                    >
                      {cell}
                    </Text>
                  </View>
                ))}
              </View>
            ))}
          </View>
        ) : (
          <Text style={styles.placeholderText}>
            The 28×28 grid will appear here after you upload an image.
          </Text>
        )}
      </View>

      {/* Bottom 20%: Prediction & Upload */}
      <View style={styles.bottomContainer}>
        {prediction !== null && (
          <View style={styles.predictionBox}>
            <Text style={styles.prediction}>Prediction: {prediction}</Text>
          </View>
        )}
        <TouchableOpacity style={styles.uploadButton} onPress={pickImage}>
          <Text style={styles.uploadButtonText}>Pick an Image</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  headerContainer: {
    flex: 0.3,
    backgroundColor: "#3498db",
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
    padding: 10,
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#fff",
  },
  modelStatus: {
    fontSize: 20,
    color: "lightgreen",
    marginVertical: 5,
  },
  inputShape: {
    fontSize: 18,
    color: "#fff",
  },
  probabilitiesContainer: {
    marginTop: 5,
    backgroundColor: "rgba(255,255,255,0.3)",
    borderRadius: 5,
    padding: 5,
    width: "90%",
    alignItems: "center",
  },
  probabilityRow: {
    fontSize: 16,
    color: "#fff",
    textAlign: "center",
  },
  gridContainer: {
    flex: 0.5,
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#ecf0f1",
    paddingHorizontal: 10,
  },
  gridBlock: {
    width: "95%",
    aspectRatio: 1, // Ensures a perfect square for the grid
    backgroundColor: "black",
  },
  gridRow: {
    flexDirection: "row",
    flex: 1,
  },
  gridCell: {
    flex: 1,
    borderWidth: 0.3,
    borderColor: "gray",
    justifyContent: "center",
    alignItems: "center",
  },
  gridCellText: {
    fontSize: 15,
    color: "white",
    fontFamily: "monospace",
    lineHeight: 17,
  },
  // Extra style for cells with value "1"
  gridCellTextOne: {
    fontSize: 17,
    fontWeight: "bold",
    color: "#f39c12",
  },
  placeholderText: {
    fontSize: 16,
    color: "#555",
    textAlign: "center",
    paddingHorizontal: 20,
  },
  bottomContainer: {
    flex: 0.2,
    backgroundColor: "#2ecc71",
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
    padding: 10,
  },
  predictionBox: {
    backgroundColor: "rgba(255,255,255,0.3)",
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 10,
    marginBottom: 10,
  },
  prediction: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#fff",
  },
  uploadButton: {
    backgroundColor: "#e74c3c",
    borderRadius: 5,
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  uploadButtonText: {
    fontSize: 18,
    color: "#fff",
    fontWeight: "bold",
  },
});
