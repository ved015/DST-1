// metro.config.js
const { getDefaultConfig } = require("expo/metro-config");

module.exports = (async () => {
  const config = await getDefaultConfig(__dirname);
  // Ensure .bin files are bundled as assets.
  config.resolver.assetExts.push("bin");
  return config;
})();
