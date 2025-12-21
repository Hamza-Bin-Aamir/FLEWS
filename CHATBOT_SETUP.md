# FLEWS Chatbot - Gemini AI Integration

## Overview
The FLEWS chatbot has been upgraded to use **Google Gemini AI** for intelligent, context-aware responses about flood safety and status updates.

## What Changed

### 1. **API Integration**
- Integrated Google Gemini Pro API for natural language understanding
- Maintains fallback to rule-based responses if API is unavailable
- Uses context-aware prompting with real-time flood data

### 2. **Files Modified**

#### `services/chatbot.py`
- Added Gemini AI integration with `google-generativeai` package
- Implemented intelligent response generation with context awareness
- Maintained backward compatibility with fallback responses
- Enhanced with real-time flood risk data analysis

#### `.env.example` 
- Added `GEMINI_API_KEY` configuration placeholder

#### `.env` (Created)
- Environment file for API keys (add your actual keys here)

#### `services/requirements.txt`
- Added `google-generativeai>=0.3.0` dependency

## Setup Instructions

### 1. Get Your Gemini API Key
1. Visit [Google AI Studio](https://ai.google.dev/)
2. Sign in with your Google account
3. Click "Get API Key"
4. Create a new API key for your project
5. Copy the API key

### 2. Configure Environment Variables
Open `.env` file and replace the placeholder:
```bash
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Install Dependencies
```bash
cd services
pip install -r requirements.txt
```

Or install just the Gemini package:
```bash
pip install google-generativeai
```

### 4. Test the Chatbot
Start the server:
```bash
cd services
python server.py
```

The chatbot will automatically:
- Use Gemini AI if API key is configured
- Fall back to rule-based responses if API key is missing or invalid
- Provide context-aware recommendations based on flood risk data

## Features

### AI-Powered Responses
- Natural language understanding of user queries
- Context-aware responses based on location and flood data
- Intelligent recommendation generation
- Emergency guidance tailored to risk levels

### Context Integration
The chatbot receives:
- User location (lat/lng coordinates)
- Real-time flood risk data from the map
- Percentage of safe, at-risk, and flooded zones
- Overall risk level assessment

### Fallback System
If Gemini API is unavailable:
- Rule-based pattern matching for common queries
- Pre-defined responses for status, emergency, and safety queries
- Context-based recommendations using flood risk data

## Example Queries

The chatbot can handle:
- **Status**: "What's the flood status in my area?"
- **Safety**: "Give me flood safety tips"
- **Emergency**: "I need emergency help"
- **Evacuation**: "How should I evacuate?"
- **General**: "What should I do to prepare for floods?"

## API Response Format

```json
{
  "response": "AI-generated or rule-based response text",
  "recommendations": [
    "Actionable recommendation 1",
    "Actionable recommendation 2",
    "..."
  ]
}
```

## Benefits of Gemini Integration

1. **Natural Conversations**: Understands varied user inputs and context
2. **Intelligent Analysis**: Interprets flood data and provides tailored advice
3. **Comprehensive Responses**: Generates detailed, helpful answers
4. **Pakistan Context**: Trained to provide Pakistan-specific emergency contacts and guidance
5. **Adaptive**: Adjusts response urgency based on actual flood risk levels

## Troubleshooting

### "Warning: Gemini API key not configured"
- Check that `GEMINI_API_KEY` is set in `.env` file
- Ensure the key is valid (not the placeholder text)
- Verify the `.env` file is in the correct directory

### API Errors
- Check your internet connection
- Verify API key is active and has quota
- Check the console for specific error messages
- System will automatically fall back to rule-based responses

### Rate Limits
- Gemini API has usage quotas (check Google AI Studio)
- If you hit rate limits, the system uses fallback responses
- Consider implementing caching for common queries

## Next Steps

Optional enhancements:
1. Implement conversation history for multi-turn dialogues
2. Add response caching to reduce API calls
3. Fine-tune system prompts for better responses
4. Add multi-language support
5. Implement streaming responses for real-time feedback

## Security Notes

- Never commit `.env` file to version control
- Keep API keys secure and private
- Regenerate keys if accidentally exposed
- Use environment-specific keys for dev/prod
