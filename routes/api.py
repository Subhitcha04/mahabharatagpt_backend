from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import get_user_model
from api.models import UserQuery
from api.serializers import UserSerializer, UserQuerySerializer

User = get_user_model()

@api_view(['POST'])
def save_query(request):
    """Saves a user query."""
    username = request.data.get('username')
    query = request.data.get('query')

    if not username or not query:
        return Response({"error": "Username and query are required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Check if the user exists or create a new user
        user, created = User.objects.get_or_create(username=username)

        # Create and save the user query
        user_query = UserQuery.objects.create(user=user, query=query)

        return Response({"message": "Query saved successfully."}, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({"error": f"Failed to save query: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
