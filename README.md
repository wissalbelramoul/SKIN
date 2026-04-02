# Skin AI Microservices

Projet microservices complet avec exigences pédagogiques.

## Architecture

- `traefik` : reverse proxy + load balancer
- `consul` : registry / service discovery
- `rabbitmq` : message queue / asynchrone
- `postgres` : base de données métier
- `auth` : authentification _JWT_ et rôles (`user` / `admin`)
- `api` : service métier REST CRUD patients + analyses + publication RabbitMQ
- `ml` : service prédiction deep learning via `/api/predict/`
- `worker` : worker asynchrone RabbitMQ pour exécuter le modèle ML
- `web` : UI statique (Nginx) et front-end JS (depuis ../frond-end)

## Points couverts (vérifiés)

1. Application REST métier (CRUD patients, analyses)
2. Service d'authentification JWT (inscription/login, token, rôles)
3. UI web + authentification / appels API
4. Communication asynchrone RabbitMQ (`skin_predictions`)
5. Service registry Consul (services s'enregistrent à démarrage)
6. Reverse proxy Traefik pour routage dynamique
7. Déploiement multi-services (containers indépendants)

## Exécution locale

1. Clone / placez dans `c:\Users\SAN\Desktop\front\microservices`
2. Démarrer:

```bash
cd c:\Users\SAN\Desktop\front\microservices
docker compose up --build -d
```

3. Vérifier état:

```bash
docker compose ps
```

4. Accéder UI:

- http://localhost/
- traefik dashboard: http://localhost:8080
- consul UI: http://localhost:8500
- rabbitmq UI: http://localhost:15672  (login `skin` / `skin_secret`)

## Endpoints utiles

### Auth
- POST `/auth/register` body JSON `{ "username": "foo", "password": "bar" }`
- POST `/auth/login` form-data `username`, `password`
- GET `/auth/me`, `/auth/verify` (token bearer)

### API métier
- GET `/api/v1/patients`
- POST `/api/v1/patients` body `{ "name": "Nom", "notes": "..." }`
- GET `/api/v1/analyses`
- POST `/api/v1/analyses` (multipart image + patient_id)

### ML
- POST `/ml/api/predict/` (multipart image)
- GET `/ml/api/health`

### Worker asynchrone
- Le `worker` consomme RabbitMQ queue `skin_predictions` et met à jour la table `analyses`.

## Tests rapides

1. Créer utilisateur et login, récupérer token.
2. Créer patient, lister patients.
3. Upload analyse, vérifier `list_analyses` en attente + worker traite et peut-être résultat.
4. Vérifier routes KS via Traefik (bande passante 80 et path rewrite).

## Remise (conforme au cahier des charges)

- Rapport technique + présentation: à compléter selon consignes enseignant.
- Déploiement multi-serveurs: architecture prête, en production déployer chaque service sur VM/container distinct.

## Remarques d'intégrité

- Le modèle ML attendu est `skin_disease_model_final.h5`; s'il n'est pas présent, le service `ml` démarre en mode `model_loaded: false`.
- `web` est servi depuis `microservices/web-ui` (UI en place). Si vous souhaitez ajouter formulaire d'authentification complet, modifiez `web-ui/index.html`.
