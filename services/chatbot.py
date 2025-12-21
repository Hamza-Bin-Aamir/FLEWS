"""
Chatbot service for FLEWS
Handles user queries about flood status, provides recommendations, and answers questions
Powered by Google Gemini AI
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import google.genai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import flood risk functions to provide real-time data
from flood_risk import get_flood_risk_areas

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here':
    os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
    GEMINI_AVAILABLE = True
else:
    GEMINI_AVAILABLE = False
    print("Warning: Gemini API key not configured. Using fallback responses.")


class FLEWSChatbot:
    """
    FLEWS Chatbot powered by Google Gemini AI that can:
    - Answer simple status questions (US17)
    - Provide dynamic recommendations based on context (US16)
    - Handle general flood safety queries
    """

    def __init__(self):
        """Initialize the chatbot with Gemini model and safety guidelines"""
        
        # Initialize Gemini client if available
        if GEMINI_AVAILABLE:
            try:
                self.client = genai.Client()
                self.model_name = 'gemini-2.5-flash'
            except Exception as e:
                print(f"Error initializing Gemini client: {e}")
                self.client = None
        else:
            self.client = None
        
        # System prompt for Gemini to guide its responses
        self.system_prompt = """You are FLEWS Assistant, an AI-powered flood early warning system chatbot for Pakistan.

Your role is to:
1. Provide accurate, timely flood safety information
2. Analyze flood risk data when available
3. Give contextual recommendations based on risk levels
4. Help users understand their flood risk and what actions to take
5. Provide emergency guidance when needed

Guidelines:
- Be concise, clear, and direct
- Use appropriate urgency levels (🚨 for high risk, ⚠️ for moderate, ✅ for low)
- Always prioritize user safety
- Reference official emergency numbers for Pakistan (1122 for emergencies)
- Provide actionable recommendations
- When flood data is available, analyze it and provide specific insights
- Keep responses focused on flood safety and Pakistan context

Emergency Contacts in Pakistan:
- Emergency Services: 1122
- NDMA Helpline: 051-9205598
- PDMA Punjab: 042-99203051
- PDMA KPK: 091-9212115
- PDMA Sindh: 021-99332003
- PDMA Balochistan: 081-9202127
"""
        self.safety_tips = {
            'general': [
                "Stay informed through official channels and weather updates",
                "Prepare an emergency kit with essentials (water, food, medicine, documents)",
                "Identify evacuation routes and safe zones in your area",
                "Keep important documents in waterproof containers",
                "Charge your phone and have a backup power source"
            ],
            'low_risk': [
                "Monitor weather forecasts and flood warnings",
                "Review your emergency preparedness plan",
                "Ensure drainage systems around your property are clear",
                "Stay alert for any changes in weather conditions",
                "Keep emergency contact numbers handy"
            ],
            'medium_risk': [
                "⚠️ Prepare to evacuate if conditions worsen",
                "Move valuable items to higher floors",
                "Fill bathtubs and containers with clean water",
                "Secure outdoor items that could be swept away",
                "Avoid unnecessary travel in affected areas",
                "Stay tuned to emergency broadcasts"
            ],
            'high_risk': [
                "🚨 EVACUATE IMMEDIATELY if ordered by authorities",
                "Move to higher ground without delay",
                "Do NOT walk or drive through flood waters",
                "Call emergency services: 1122 if in danger",
                "Take only essential items (documents, medicine, water)",
                "Follow official evacuation routes",
                "Do NOT return until authorities declare it safe"
            ],
            'evacuation': [
                "🚨 Follow official evacuation orders immediately",
                "Turn off utilities (gas, electricity, water) if time permits",
                "Take emergency kit, documents, medicine, and essentials",
                "Wear appropriate clothing and sturdy shoes",
                "Avoid walking/driving through water (6 inches can knock you down)",
                "Head to designated evacuation centers or higher ground",
                "Inform family/friends of your evacuation plan",
                "Do NOT return until officially declared safe"
            ]
        }

        self.emergency_contacts = {
            "Emergency Services": "1122",
            "NDMA Helpline": "051-9205598",
            "PDMA Punjab": "042-99203051",
            "PDMA KPK": "091-9212115",
            "PDMA Sindh": "021-99332003",
            "PDMA Balochistan": "081-9202127"
        }

    def process_message(self, message: str, context: Optional[Dict] = None) -> Dict:
        """
        Process user message and generate response using Gemini AI
        
        Args:
            message: User's message
            context: Optional context including location and flood data
            
        Returns:
            Dictionary with response and recommendations
        """
        
        # Extract context information
        location = context.get('location') if context else None
        flood_data = context.get('flood_data') if context else None
        
        # Build context for the AI
        context_str = self._build_context_string(location, flood_data)
        
        # If Gemini is available, use it
        if GEMINI_AVAILABLE and self.client:
            try:
                return self._get_gemini_response(message, context_str, location, flood_data)
            except Exception as e:
                print(f"Error using Gemini API: {e}")
                # Fall back to rule-based system
                return self._get_fallback_response(message, location, flood_data)
        else:
            # Use fallback rule-based system
            return self._get_fallback_response(message, location, flood_data)
    
    def _build_context_string(self, location: Optional[List], flood_data: Optional[Dict]) -> str:
        """Build a context string for the AI based on available data"""
        context_parts = []
        
        if location and len(location) == 2:
            lng, lat = location
            context_parts.append(f"User location: Latitude {lat}, Longitude {lng}")
            
            # Get flood risk data for the location
            try:
                tiles = get_flood_risk_areas(lat - 0.05, lat + 0.05, lng - 0.05, lng + 0.05)
                
                if tiles:
                    risk_counts = {'Safe': 0, 'At Risk': 0, 'Flooded': 0}
                    for tile in tiles:
                        risk_counts[tile['status']] = risk_counts.get(tile['status'], 0) + 1
                    
                    total = sum(risk_counts.values())
                    context_parts.append(f"\nFlood Risk Data:")
                    context_parts.append(f"- Safe zones: {risk_counts['Safe']} ({risk_counts['Safe']/total*100:.1f}%)")
                    context_parts.append(f"- At-risk zones: {risk_counts['At Risk']} ({risk_counts['At Risk']/total*100:.1f}%)")
                    context_parts.append(f"- Flooded zones: {risk_counts['Flooded']} ({risk_counts['Flooded']/total*100:.1f}%)")
                    
                    # Determine overall risk level
                    if risk_counts['Flooded'] > total * 0.3:
                        context_parts.append(f"\nOVERALL RISK LEVEL: HIGH (Critical flooding detected)")
                    elif risk_counts['At Risk'] > total * 0.3 or risk_counts['Flooded'] > 0:
                        context_parts.append(f"\nOVERALL RISK LEVEL: MODERATE (Some flooding or high risk areas)")
                    else:
                        context_parts.append(f"\nOVERALL RISK LEVEL: LOW (Minimal flood risk)")
            except Exception as e:
                print(f"Error getting flood data: {e}")
        
        if flood_data:
            context_parts.append(f"\nAdditional flood data provided: {flood_data}")
        
        if not context_parts:
            context_parts.append("No specific location or flood data available")
        
        return "\n".join(context_parts)
    
    def _get_gemini_response(self, message: str, context_str: str, 
                            location: Optional[List], flood_data: Optional[Dict]) -> Dict:
        """Get response from Gemini AI"""
        
        # Construct the full prompt
        full_prompt = f"""{self.system_prompt}

CURRENT CONTEXT:
{context_str}

USER MESSAGE: {message}

Please provide:
1. A clear, helpful response to the user's question
2. Specific actionable recommendations based on the flood risk level (if applicable)

Format your response naturally and conversationally. If recommending actions, list them clearly."""

        # Generate response using new client pattern
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt
        )
        response_text = response.text
        
        # Extract recommendations if present
        recommendations = self._extract_recommendations(response_text, location, flood_data)
        
        return {
            'response': response_text,
            'recommendations': recommendations
        }
    
    def _extract_recommendations(self, response_text: str, 
                                 location: Optional[List], 
                                 flood_data: Optional[Dict]) -> List[str]:
        """Extract actionable recommendations from the response"""
        recommendations = []
        
        # Look for bullet points or numbered lists in the response
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            # Match bullet points or numbered items
            if line and (line.startswith('•') or line.startswith('-') or 
                        line.startswith('*') or re.match(r'^\d+\.', line)):
                # Clean up the recommendation
                rec = re.sub(r'^[•\-*\d\.]+\s*', '', line)
                if rec and len(rec) > 10:  # Only add substantial recommendations
                    recommendations.append(rec)
        
        # If no recommendations found in response, add context-based ones
        if not recommendations:
            recommendations = self._get_context_recommendations(location, flood_data)
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def _get_context_recommendations(self, location: Optional[List], 
                                     flood_data: Optional[Dict]) -> List[str]:
        """Get context-appropriate recommendations based on risk level"""
    def _get_context_recommendations(self, location: Optional[List], 
                                     flood_data: Optional[Dict]) -> List[str]:
        """Get context-appropriate recommendations based on risk level"""
        
        if not location or len(location) != 2:
            return [
                "Stay informed through official channels and weather updates",
                "Prepare an emergency kit with essentials",
                "Identify evacuation routes in your area",
                "Keep emergency contact numbers handy"
            ]
        
        try:
            lng, lat = location
            tiles = get_flood_risk_areas(lat - 0.05, lat + 0.05, lng - 0.05, lng + 0.05)
            
            if tiles:
                risk_counts = {'Safe': 0, 'At Risk': 0, 'Flooded': 0}
                for tile in tiles:
                    risk_counts[tile['status']] = risk_counts.get(tile['status'], 0) + 1
                
                total = sum(risk_counts.values())
                
                # High risk recommendations
                if risk_counts['Flooded'] > total * 0.3:
                    return [
                        "🚨 EVACUATE IMMEDIATELY if ordered by authorities",
                        "Move to higher ground without delay",
                        "Do NOT walk or drive through flood waters",
                        "Call emergency services: 1122 if in danger",
                        "Take only essential items (documents, medicine, water)",
                        "Follow official evacuation routes"
                    ]
                # Moderate risk recommendations
                elif risk_counts['At Risk'] > total * 0.3 or risk_counts['Flooded'] > 0:
                    return [
                        "⚠️ Prepare to evacuate if conditions worsen",
                        "Move valuable items to higher floors",
                        "Fill containers with clean water",
                        "Secure outdoor items that could be swept away",
                        "Avoid unnecessary travel in affected areas",
                        "Stay tuned to emergency broadcasts"
                    ]
                # Low risk recommendations
                else:
                    return [
                        "Monitor weather forecasts and flood warnings",
                        "Review your emergency preparedness plan",
                        "Ensure drainage systems are clear",
                        "Stay alert for any changes in weather conditions",
                        "Keep emergency contact numbers handy"
                    ]
        except Exception as e:
            print(f"Error getting recommendations: {e}")
        
        # Default recommendations
        return [
            "Stay informed through official channels",
            "Prepare an emergency kit",
            "Identify evacuation routes",
            "Keep important documents safe"
        ]
    
    def _get_fallback_response(self, message: str, location: Optional[List], 
                               flood_data: Optional[Dict]) -> Dict:
        """Fallback rule-based response when Gemini is not available"""
        
        message_lower = message.lower()
        
        # Status query
        if any(word in message_lower for word in ['status', 'safe', 'flood', 'risk', 'area']):
            return self._fallback_status(location, flood_data)
        
        # Emergency query
        elif any(word in message_lower for word in ['emergency', 'help', 'danger', 'urgent']):
            return self._fallback_emergency()
        
        # Safety tips query
        elif any(word in message_lower for word in ['tip', 'advice', 'recommend', 'what should', 'how to']):
            return self._fallback_safety(location, flood_data)
        
        # Greeting
        elif any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return self._fallback_greeting()
        
        # Default
        else:
            return self._fallback_general()
    
    def _fallback_status(self, location: Optional[List], flood_data: Optional[Dict]) -> Dict:
        """Fallback status response"""
        
        if location and len(location) == 2:
            lng, lat = location
            try:
                tiles = get_flood_risk_areas(lat - 0.05, lat + 0.05, lng - 0.05, lng + 0.05)
                
                if tiles:
                    risk_counts = {'Safe': 0, 'At Risk': 0, 'Flooded': 0}
                    for tile in tiles:
                        risk_counts[tile['status']] = risk_counts.get(tile['status'], 0) + 1
                    
                    total = sum(risk_counts.values())
                    
                    if risk_counts['Flooded'] > total * 0.3:
                        status_msg = "🚨 **HIGH RISK**: Your area is experiencing significant flooding."
                    elif risk_counts['At Risk'] > total * 0.3 or risk_counts['Flooded'] > 0:
                        status_msg = "⚠️ **MODERATE RISK**: Your area is at risk of flooding."
                    else:
                        status_msg = "✅ **LOW RISK**: Your area appears relatively safe."
                    
                    response = f"{status_msg}\n\nBased on current data:\n• Safe zones: {risk_counts['Safe']}\n• At-risk zones: {risk_counts['At Risk']}\n• Flooded zones: {risk_counts['Flooded']}"
                    
                    return {
                        'response': response,
                        'recommendations': self._get_context_recommendations(location, flood_data)
                    }
            except Exception as e:
                print(f"Error in fallback status: {e}")
        
        return {
            'response': "I can help you check flood status. Please enable location services or search for a specific area on the map.",
            'recommendations': self._get_context_recommendations(None, None)
        }
    
    def _fallback_emergency(self) -> Dict:
        """Fallback emergency response"""
        response = """🚨 **EMERGENCY RESPONSE**

If you are in immediate danger:
1. **Call Emergency Services: 1122**
2. Move to higher ground immediately
3. Do NOT walk or drive through flood water
4. Signal for help from a safe location

**Emergency Contacts:**
• Emergency Services: 1122
• NDMA Helpline: 051-9205598
• PDMA Punjab: 042-99203051"""
        
        return {
            'response': response,
            'recommendations': [
                "Call 1122 immediately if in danger",
                "Move to higher ground",
                "Do NOT enter flood waters",
                "Stay where rescuers can see you",
                "Keep phone charged for communication"
            ]
        }
    
    def _fallback_safety(self, location: Optional[List], flood_data: Optional[Dict]) -> Dict:
        """Fallback safety tips response"""
        response = "Here are important flood safety recommendations:"
        
        return {
            'response': response,
            'recommendations': self._get_context_recommendations(location, flood_data)
        }
    
    def _fallback_greeting(self) -> Dict:
        """Fallback greeting response"""
        response = """Hello! 👋 I'm your FLEWS assistant, here to help you stay safe during floods.

I can provide:
• Real-time flood status updates
• Safety recommendations
• Emergency guidance
• Evacuation information

What would you like to know?"""
        
        return {
            'response': response,
            'recommendations': []
        }
    
    def _fallback_general(self) -> Dict:
        """Fallback general response"""
        response = """I'm here to help with flood-related information. You can ask me:

• About flood status in your area
• For safety tips and recommendations
• About emergency procedures
• For evacuation guidance

Try asking: "What's the flood status?" or "Give me safety tips\""""
        
        return {
            'response': response,
            'recommendations': self._get_context_recommendations(None, None)
        }


# Create a singleton instance
chatbot = FLEWSChatbot()


def get_chat_response(message: str, context: Optional[Dict] = None) -> Dict:
    """
    Get chatbot response for a user message
    
    Args:
        message: User's message
        context: Optional context with location and flood data
        
    Returns:
        Dictionary with response and recommendations
    """
    return chatbot.process_message(message, context)
