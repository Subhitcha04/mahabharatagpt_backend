from django.contrib import admin
from django.utils.safestring import mark_safe
import json
from .models import (
    Character, StoryOrEvent, SceneOrIncident, PlaceOrLocation,
    ObjectOrArtifact, ThemeOrMoral, MythologySystem, MythologyEra,
    CreatureOrSpecies, ProphecyOrFate, Comparison, CulturalOrHistorical,
    RiddleOrPuzzle, AppUser, UserQuery
)

class ImagePreviewMixin:
    """Mixin to display an image preview in Django Admin."""
    
    def image_preview(self, obj):
        """Displays image preview in the admin panel."""
        if obj.image:
            return mark_safe(f'<img src="{obj.image.url}" width="150" height="150" style="object-fit: cover;" />')
        return "No Image"

    image_preview.short_description = "Image Preview"

    def format_json_field(self, obj, field_name):
        """Formats JSON data for better display in Django Admin."""
        field_value = getattr(obj, field_name, None)
        if isinstance(field_value, (dict, list)):
            return mark_safe(f'<pre>{json.dumps(field_value, indent=2, ensure_ascii=False)}</pre>')
        return field_value or "No Data"

    def formatted_queries(self, obj):
        """Formats the 'queries' field."""
        return self.format_json_field(obj, "queries")

    def formatted_answers(self, obj):
        """Formats the 'answers' field."""
        return self.format_json_field(obj, "answers")

    formatted_queries.short_description = "Queries"
    formatted_answers.short_description = "Answers"


class BaseAdmin(admin.ModelAdmin, ImagePreviewMixin):
    """Base admin class for shared functionalities."""
    
    readonly_fields = ('image_preview', 'formatted_queries', 'formatted_answers')
    search_fields = ('name', 'title', 'scene_title', 'place_name', 'object_name', 'theme', 'system_name', 'era_name', 'species_name', 'prophecy_title', 'comparison_title', 'culture_event', 'riddle')
    ordering = ('id',)


@admin.register(Character)
class CharacterAdmin(BaseAdmin):
    list_display = ('name', 'manual_entry', 'image_preview')
    fields = ('name', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(StoryOrEvent)
class StoryOrEventAdmin(BaseAdmin):
    list_display = ('title', 'manual_entry', 'image_preview')
    fields = ('title', 'formatted_queries', 'formatted_answers', 'source', 'image', 'image_preview', 'manual_entry')


@admin.register(SceneOrIncident)
class SceneOrIncidentAdmin(BaseAdmin):
    list_display = ('scene_title', 'manual_entry', 'image_preview')
    fields = ('scene_title', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(PlaceOrLocation)
class PlaceOrLocationAdmin(BaseAdmin):
    list_display = ('place_name', 'manual_entry', 'image_preview')
    fields = ('place_name', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(ObjectOrArtifact)
class ObjectOrArtifactAdmin(BaseAdmin):
    list_display = ('object_name', 'manual_entry', 'image_preview')
    fields = ('object_name', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(ThemeOrMoral)
class ThemeOrMoralAdmin(BaseAdmin):
    list_display = ('theme', 'manual_entry', 'image_preview')
    fields = ('theme', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(MythologySystem)
class MythologySystemAdmin(BaseAdmin):
    list_display = ('system_name', 'manual_entry', 'image_preview')
    fields = ('system_name', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(MythologyEra)
class MythologyEraAdmin(BaseAdmin):
    list_display = ('era_name', 'manual_entry', 'image_preview')
    fields = ('era_name', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(CreatureOrSpecies)
class CreatureOrSpeciesAdmin(BaseAdmin):
    list_display = ('species_name', 'manual_entry', 'image_preview')
    fields = ('species_name', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(ProphecyOrFate)
class ProphecyOrFateAdmin(BaseAdmin):
    list_display = ('prophecy_title', 'manual_entry', 'image_preview')
    fields = ('prophecy_title', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(Comparison)
class ComparisonAdmin(BaseAdmin):
    list_display = ('comparison_title', 'manual_entry', 'image_preview')
    fields = ('comparison_title', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(CulturalOrHistorical)
class CulturalOrHistoricalAdmin(BaseAdmin):
    list_display = ('culture_event', 'manual_entry', 'image_preview')
    fields = ('culture_event', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(RiddleOrPuzzle)
class RiddleOrPuzzleAdmin(BaseAdmin):
    list_display = ('riddle', 'manual_entry', 'image_preview')
    fields = ('riddle', 'formatted_queries', 'formatted_answers', 'image', 'image_preview', 'manual_entry')


@admin.register(AppUser)
class AppUserAdmin(admin.ModelAdmin):
    """Admin panel for AppUser model."""
    
    list_display = ('id', 'username', 'email', 'created_at')
    search_fields = ('username', 'email')
    ordering = ('-created_at',)
    list_filter = ('created_at',)


@admin.register(UserQuery)
class UserQueryAdmin(admin.ModelAdmin):
    """Admin panel for UserQuery model."""
    
    list_display = ('id', 'get_username', 'query', 'created_at')
    search_fields = ('user__username', 'query')
    list_filter = ('created_at',)
    ordering = ('-created_at',)

    def get_username(self, obj):
        """Displays the username of the related user."""
        return obj.user.username if obj.user else "Anonymous"
    
    get_username.admin_order_field = 'user'
    get_username.short_description = 'Username'
