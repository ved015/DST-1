import * as tf from "@tensorflow/tfjs-react-native";
import * as FileSystem from "expo-file-system";

if (!global.fetch) {
  global.fetch = require("node-fetch");
}

// Override file storage methods to use expo-file-system
tf.io.getModelArtifactsInfoForJSON = async (modelJson) => {
  const modelArtifacts = await tf.io.browserHTTPRequest(modelJson).load();
  return modelArtifacts;
};

export default tf;
