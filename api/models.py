from django.db import models
import uuid

class Character(models.Model):
    name = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='characters/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.name


class StoryOrEvent(models.Model):
    title = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    source = models.CharField(max_length=255)
    image = models.ImageField(upload_to='stories/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.title


class SceneOrIncident(models.Model):
    scene_title = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='scenes/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.scene_title


class PlaceOrLocation(models.Model):
    place_name = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='places/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.place_name


class ObjectOrArtifact(models.Model):
    object_name = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='objects/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.object_name


class ThemeOrMoral(models.Model):
    theme = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='themes/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.theme


class MythologySystem(models.Model):
    system_name = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='mythology_systems/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.system_name


class MythologyEra(models.Model):
    era_name = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='mythology_eras/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.era_name


class CreatureOrSpecies(models.Model):
    species_name = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='creatures/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.species_name


class ProphecyOrFate(models.Model):
    prophecy_title = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='prophecies/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.prophecy_title


class Comparison(models.Model):
    comparison_title = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='comparisons/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.comparison_title


class CulturalOrHistorical(models.Model):
    culture_event = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='cultural_events/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.culture_event


class RiddleOrPuzzle(models.Model):
    riddle = models.CharField(max_length=255)
    queries = models.JSONField()
    answers = models.JSONField()
    image = models.ImageField(upload_to='riddles/', blank=True, null=True)
    manual_entry = models.BooleanField(default=False)

    def __str__(self):
        return self.riddle

class AppUser(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)  # UUID for AppUser
    username = models.CharField(max_length=255, unique=True)
    email = models.EmailField(max_length=255, unique=True)  # Unique email field
    password = models.CharField(max_length=255)  # Store hashed passwords
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username


class UserQuery(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)  # UUID for UserQuery
    user = models.ForeignKey(
        AppUser,
        on_delete=models.CASCADE,
        null=False,
        blank=False
    )
    query = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.query[:50]}"
