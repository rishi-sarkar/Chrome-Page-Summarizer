// const path = require('path');
// const webpack = require('webpack');

// module.exports = {
//   mode: 'development', // Change to 'production' for production builds.
//   entry: {
//     popup: path.resolve(__dirname, 'src', 'popup.js'),
//     contentScript: path.resolve(__dirname, 'src', 'contentScript.js')
//   },
//   output: {
//     path: path.resolve(__dirname, 'dist'),
//     filename: '[name].bundle.js'
//   },
//   module: {
//     rules: [
//       // Transpile modern JavaScript.
//       {
//         test: /\.js$/,
//         exclude: /node_modules/,
//         use: {
//           loader: 'babel-loader',
//           options: {
//             presets: ['@babel/preset-env']
//           }
//         }
//       },
//       // Handle WebAssembly files if needed by tokenizers.
//       {
//         test: /\.wasm$/,
//         type: "webassembly/async"
//       }
//     ]
//   },
//   externals: {
//     // Do not bundle the tokenizers module.
//     tokenizers: 'tokenizers'
//   },
//   experiments: {
//     asyncWebAssembly: true,
//   },
//   plugins: [
//     // Optionally, add any plugins you need.
//     new webpack.ProvidePlugin({
//       // Define global variables if needed.
//     })
//   ]
// };
