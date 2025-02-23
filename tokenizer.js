import { Tokenizer } from 'tokenizers';  // or the appropriate import if installed from GitHub

let tokenizer;

export async function loadTokenizer() {
  // Adjust the file path if needed. This file should be part of your extension's resources.
  const tokenizerFileUrl = chrome.runtime.getURL('tokenizer.json');
  tokenizer = await Tokenizer.fromFile(tokenizerFileUrl);
  return tokenizer;
}

export function tokenizeText(text) {
  if (!tokenizer) {
    throw new Error("Tokenizer not loaded");
  }
  // Tokenizer.encode returns an object with token IDs.
  const encoded = tokenizer.encode(text);
  return encoded.ids;
}

export function decodeTokens(tokenIds) {
  if (!tokenizer) {
    throw new Error("Tokenizer not loaded");
  }
  return tokenizer.decode(tokenIds, true);
}
