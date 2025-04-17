from rest_framework import serializers
from api.models import (
    AppUser, UserQuery, Character, StoryOrEvent, SceneOrIncident, PlaceOrLocation, 
    ObjectOrArtifact, ThemeOrMoral, MythologySystem, MythologyEra, CreatureOrSpecies, 
    ProphecyOrFate, Comparison, CulturalOrHistorical, RiddleOrPuzzle
)

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = AppUser
        fields = ['id', 'username', 'email', 'password']
        extra_kwargs = {'password': {'write_only': True}}  # Hide password from response


class UserQuerySerializer(serializers.ModelSerializer):
    user = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = UserQuery
        fields = ['id', 'user', 'query', 'created_at']


class JSONSerializerField(serializers.JSONField):
    """Custom serializer for handling JSON fields."""
    def to_representation(self, value):
        return super().to_representation(value)

    def to_internal_value(self, data):
        return super().to_internal_value(data)


class BaseSerializer(serializers.ModelSerializer):
    queries = JSONSerializerField()
    answers = JSONSerializerField()


class CharacterSerializer(BaseSerializer):
    class Meta:
        model = Character
        fields = '__all__'


class StoryOrEventSerializer(BaseSerializer):
    class Meta:
        model = StoryOrEvent
        fields = '__all__'


class SceneOrIncidentSerializer(BaseSerializer):
    class Meta:
        model = SceneOrIncident
        fields = '__all__'


class PlaceOrLocationSerializer(BaseSerializer):
    class Meta:
        model = PlaceOrLocation
        fields = '__all__'


class ObjectOrArtifactSerializer(BaseSerializer):
    class Meta:
        model = ObjectOrArtifact
        fields = '__all__'


class ThemeOrMoralSerializer(BaseSerializer):
    class Meta:
        model = ThemeOrMoral
        fields = '__all__'


class MythologySystemSerializer(BaseSerializer):
    class Meta:
        model = MythologySystem
        fields = '__all__'


class MythologyEraSerializer(BaseSerializer):
    class Meta:
        model = MythologyEra
        fields = '__all__'


class CreatureOrSpeciesSerializer(BaseSerializer):
    class Meta:
        model = CreatureOrSpecies
        fields = '__all__'


class ProphecyOrFateSerializer(BaseSerializer):
    class Meta:
        model = ProphecyOrFate
        fields = '__all__'


class ComparisonSerializer(BaseSerializer):
    class Meta:
        model = Comparison
        fields = '__all__'


class CulturalOrHistoricalSerializer(BaseSerializer):
    class Meta:
        model = CulturalOrHistorical
        fields = '__all__'


class RiddleOrPuzzleSerializer(BaseSerializer):
    class Meta:
        model = RiddleOrPuzzle
        fields = '__all__'
