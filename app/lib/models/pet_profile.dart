/// 宠物信息
class PetProfile {
  final String id;
  String name;
  String species; // cat, dog
  int age;
  String breed;
  String notes;

  PetProfile({
    required this.id,
    required this.name,
    this.species = 'cat',
    this.age = 1,
    this.breed = '',
    this.notes = '',
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'species': species,
    'age': age,
    'breed': breed,
    'notes': notes,
  };

  factory PetProfile.fromJson(Map<String, dynamic> json) => PetProfile(
    id: json['id'] as String,
    name: json['name'] as String,
    species: json['species'] as String? ?? 'cat',
    age: json['age'] as int? ?? 1,
    breed: json['breed'] as String? ?? '',
    notes: json['notes'] as String? ?? '',
  );
}
