# Infomaniak AI Integration for Cheshire Cat

This plugin integrates Infomaniak AI services with the Cheshire Cat AI framework.

## Features

- Chat completions with Infomaniak AI models
- Document and query embeddings

## Configuration

To use this plugin, you need:

1. An Infomaniak account with AI services enabled
2. An API key from Infomaniak
3. Your Infomaniak AI Product ID (found in your Infomaniak AI dashboard)

Configure the plugin in the Cheshire Cat settings with:
- Your Infomaniak API key
- Your Product ID
- The model you want to use for chat (default: mixtral)
- The model you want to use for embeddings (default: text-embedding-ada-002)
- Temperature and other optional parameters

## Models Available

The following models are available through Infomaniak's OpenAI-compatible API:
- llama3
- mixtral

### Embedding Models
- bge_multilingual_gemma2
- all_minilm_l12_v2

## API Documentation

For more information about the APIs, visit:
- Chat: https://developer.infomaniak.com/docs/api/post/1/ai/{product_id}/openai/chat/completions
- Embeddings: https://developer.infomaniak.com/docs/api/post/1/ai/{product_id}/openai/v1/embeddings

## Support

For issues and feature requests, please visit the [GitHub repository](https://github.com/Lukid/infomaniak_integration).