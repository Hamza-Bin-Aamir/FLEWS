"""
Chatbot service for FLEWS
Handles user queries about flood status, provides recommendations, and answers questions
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import flood risk functions to provide real-time data
from flood_risk import get_flood_risk_areas


class FLEWSChatbot:
    """
    FLEWS Chatbot that can:
    - Answer simple status questions (US17)
    - Provide dynamic recommendations based on context (US16)
    - Handle general flood safety queries
    """

    def __init__(self):
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
        Process user message and generate response
        
        Args:
            message: User's message
            context: Optional context including location and flood data
            
        Returns:
            Dictionary with response and recommendations
        """
        message_lower = message.lower()
        
        # Extract context information
        location = context.get('location') if context else None
        flood_data = context.get('flood_data') if context else None
        
        # Determine response type based on message content
        response_type = self._classify_query(message_lower)
        
        if response_type == 'status':
            return self._handle_status_query(message_lower, location, flood_data)
        elif response_type == 'safety':
            return self._handle_safety_query(message_lower, location, flood_data)
        elif response_type == 'emergency':
            return self._handle_emergency_query(message_lower)
        elif response_type == 'evacuation':
            return self._handle_evacuation_query(message_lower)
        elif response_type == 'help':
            return self._handle_help_query()
        elif response_type == 'greeting':
            return self._handle_greeting()
        else:
            return self._handle_general_query(message_lower, location, flood_data)

    def _classify_query(self, message: str) -> str:
        """Classify the type of user query"""
        
        # Status queries (US17)
        status_keywords = ['status', 'safe', 'flood', 'risk', 'danger', 'area', 'region', 'current', 'now', 'today']
        if any(keyword in message for keyword in status_keywords) and not any(word in message for word in ['what should', 'how to', 'help me']):
            return 'status'
        
        # Emergency queries
        emergency_keywords = ['emergency', 'help', 'danger', 'urgent', 'stuck', 'trapped', 'rescue']
        if any(keyword in message for keyword in emergency_keywords):
            return 'emergency'
        
        # Evacuation queries
        evacuation_keywords = ['evacuate', 'evacuation', 'leave', 'escape', 'get out']
        if any(keyword in message for keyword in evacuation_keywords):
            return 'evacuation'
        
        # Safety/tips queries
        safety_keywords = ['safe', 'tip', 'advice', 'recommend', 'what should', 'how to', 'prepare', 'protect']
        if any(keyword in message for keyword in safety_keywords):
            return 'safety'
        
        # Help queries
        help_keywords = ['help', 'can you', 'what can', 'how do', 'guide']
        if any(keyword in message for keyword in help_keywords):
            return 'help'
        
        # Greeting
        greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(keyword in message for keyword in greeting_keywords):
            return 'greeting'
        
        return 'general'

    def _handle_status_query(self, message: str, location: Optional[List], flood_data: Optional[Dict]) -> Dict:
        """Handle status queries (US17)"""
        
        # Try to get real-time flood status
        if location and len(location) == 2:
            lng, lat = location
            # Get flood risk for the specific location
            tiles = get_flood_risk_areas(lat - 0.05, lat + 0.05, lng - 0.05, lng + 0.05)
            
            if tiles:
                # Analyze the tiles around the location
                risk_counts = {'Safe': 0, 'At Risk': 0, 'Flooded': 0}
                for tile in tiles:
                    risk_counts[tile['status']] = risk_counts.get(tile['status'], 0) + 1
                
                total = sum(risk_counts.values())
                
                # Determine overall risk level
                if risk_counts['Flooded'] > total * 0.3:
                    risk_level = 'high_risk'
                    status_msg = "🚨 **HIGH RISK**: Your area is experiencing significant flooding. Immediate action may be required."
                elif risk_counts['At Risk'] > total * 0.3 or risk_counts['Flooded'] > 0:
                    risk_level = 'medium_risk'
                    status_msg = "⚠️ **MODERATE RISK**: Your area is at risk of flooding. Stay alert and prepared."
                else:
                    risk_level = 'low_risk'
                    status_msg = "✅ **LOW RISK**: Your area appears relatively safe at the moment, but continue monitoring conditions."
                
                recommendations = self.safety_tips[risk_level]
                
                return {
                    'response': f"{status_msg}\n\nBased on current data:\n• Safe zones: {risk_counts['Safe']}\n• At-risk zones: {risk_counts['At Risk']}\n• Flooded zones: {risk_counts['Flooded']}\n\nPlease follow the recommendations below and stay informed.",
                    'recommendations': recommendations
                }
        
        # Default response when location is not available
        return {
            'response': "I can help you check flood status, but I need your location data. You can:\n\n• Search for a specific area on the map\n• Ask about a particular city or region\n• Check the color-coded map for risk levels:\n  🟢 Green = Safe\n  🟡 Yellow = At Risk\n  🔴 Red = Flooded\n\nWhat area would you like to check?",
            'recommendations': self.safety_tips['general']
        }

    def _handle_safety_query(self, message: str, location: Optional[List], flood_data: Optional[Dict]) -> Dict:
        """Handle safety and recommendation queries (US16)"""
        
        # Determine context-specific recommendations
        risk_level = 'general'
        
        if flood_data or location:
            # If we have flood data, analyze the risk
            if location and len(location) == 2:
                lng, lat = location
                tiles = get_flood_risk_areas(lat - 0.05, lat + 0.05, lng - 0.05, lng + 0.05)
                
                if tiles:
                    flooded = sum(1 for t in tiles if t['status'] == 'Flooded')
                    at_risk = sum(1 for t in tiles if t['status'] == 'At Risk')
                    total = len(tiles)
                    
                    if flooded > total * 0.3:
                        risk_level = 'high_risk'
                    elif at_risk > total * 0.3 or flooded > 0:
                        risk_level = 'medium_risk'
                    else:
                        risk_level = 'low_risk'
        
        recommendations = self.safety_tips[risk_level]
        
        response = f"Here are important flood safety recommendations for your situation:\n\n"
        
        if risk_level == 'high_risk':
            response += "⚠️ **CRITICAL**: Due to high flood risk in your area, immediate precautions are necessary."
        elif risk_level == 'medium_risk':
            response += "⚠️ **CAUTION**: Your area has moderate flood risk. Stay prepared and vigilant."
        else:
            response += "These general safety tips will help you stay prepared:"
        
        return {
            'response': response,
            'recommendations': recommendations
        }

    def _handle_emergency_query(self, message: str) -> Dict:
        """Handle emergency queries"""
        
        response = "🚨 **EMERGENCY RESPONSE**\n\n"
        response += "If you are in immediate danger:\n\n"
        response += "1. **Call Emergency Services: 1122** (Pakistan Emergency Helpline)\n"
        response += "2. Move to higher ground immediately\n"
        response += "3. Do NOT attempt to walk or drive through flood water\n"
        response += "4. Signal for help from a safe location\n\n"
        response += "**Important Emergency Contacts:**\n"
        
        for service, number in self.emergency_contacts.items():
            response += f"• {service}: {number}\n"
        
        return {
            'response': response,
            'recommendations': self.safety_tips['high_risk']
        }

    def _handle_evacuation_query(self, message: str) -> Dict:
        """Handle evacuation queries"""
        
        response = "🚨 **EVACUATION GUIDANCE**\n\n"
        response += "If you need to evacuate due to flooding:\n\n"
        response += "1. **Follow all official evacuation orders immediately**\n"
        response += "2. Listen to local authorities and emergency broadcasts\n"
        response += "3. Take only essential items that you can carry\n"
        response += "4. Turn off utilities if you have time\n"
        response += "5. Use designated evacuation routes\n"
        response += "6. Head to official evacuation centers or higher ground\n\n"
        response += "**Emergency Helpline: 1122**"
        
        return {
            'response': response,
            'recommendations': self.safety_tips['evacuation']
        }

    def _handle_help_query(self) -> Dict:
        """Handle help queries"""
        
        response = "I'm your FLEWS (Flood Early Warning System) assistant! Here's how I can help:\n\n"
        response += "**🔍 Status Checks**\n"
        response += "• \"What's the flood status in my area?\"\n"
        response += "• \"Is my area safe?\"\n"
        response += "• \"Check flood risk near me\"\n\n"
        response += "**💡 Safety & Recommendations**\n"
        response += "• \"Give me flood safety tips\"\n"
        response += "• \"How should I prepare?\"\n"
        response += "• \"What should I do now?\"\n\n"
        response += "**🚨 Emergency Help**\n"
        response += "• \"Emergency contacts\"\n"
        response += "• \"How to evacuate?\"\n"
        response += "• \"I need immediate help\"\n\n"
        response += "Just ask me anything about flood safety and status!"
        
        return {
            'response': response,
            'recommendations': []
        }

    def _handle_greeting(self) -> Dict:
        """Handle greeting messages"""
        
        response = "Hello! 👋 I'm your FLEWS assistant, here to help you stay safe during floods.\n\n"
        response += "I can provide:\n"
        response += "• Real-time flood status updates\n"
        response += "• Safety recommendations\n"
        response += "• Emergency guidance\n"
        response += "• Evacuation information\n\n"
        response += "What would you like to know?"
        
        return {
            'response': response,
            'recommendations': []
        }

    def _handle_general_query(self, message: str, location: Optional[List], flood_data: Optional[Dict]) -> Dict:
        """Handle general queries"""
        
        response = "I'm here to help with flood-related information. You can ask me:\n\n"
        response += "• About flood status in your area\n"
        response += "• For safety tips and recommendations\n"
        response += "• About emergency procedures\n"
        response += "• For evacuation guidance\n\n"
        response += "Try asking: \"What's the flood status?\" or \"Give me safety tips\""
        
        return {
            'response': response,
            'recommendations': self.safety_tips['general']
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
