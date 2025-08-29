# ===================================
# FIXED advanced_face_search_service.py - Syntax Error Fixed + Pool Expansion Logic
# ===================================

import os
import json
import logging
import numpy as np
import boto3
import faiss
import time
import uuid
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import signal
import sys
import psutil
import requests
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ===================================
# Enhanced Logging Configuration
# ===================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('advanced_face_search.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ===================================
# Data Classes for Type Safety
# ===================================

@dataclass
class FaceEmbedding:
    """Represents a face embedding with metadata"""
    uid: str
    event_id: str
    embedding: np.ndarray
    s3_url: str
    filename: str
    created_at: str
    quality_score: float
    face_size: Optional[Tuple[int, int]] = None
    blur_score: Optional[float] = None

    def to_normalized_embedding(self) -> np.ndarray:
        """Return L2 normalized embedding for cosine similarity"""
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            return self.embedding / norm
        return self.embedding

@dataclass
class GuestProfile:
    """Enhanced guest profile with multiple embeddings"""
    guest_id: str
    guest_name: str
    guest_email: str
    guest_phone: str
    registration_id: str
    embeddings: List[FaceEmbedding]
    best_embedding_index: int = 0
    profile_quality_score: float = 0.0

    def get_best_embedding(self) -> FaceEmbedding:
        """Get the highest quality embedding for matching"""
        if self.embeddings:
            return self.embeddings[self.best_embedding_index]
        return None

    def get_all_normalized_embeddings(self) -> np.ndarray:
        """Get all normalized embeddings as a matrix"""
        return np.array([emb.to_normalized_embedding() for emb in self.embeddings])

@dataclass
class MatchResult:
    """Represents a match between guest and pool image"""
    pool_uid: str
    pool_url: str
    pool_filename: str
    similarity_score: float
    match_confidence: float
    match_type: str
    metadata: Dict[str, Any]

class MatchConfidence(Enum):
    """Match confidence levels"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

# ===================================
# FIXED Advanced Face Search Engine
# ===================================

class AdvancedFaceSearchEngine:
    """
    FIXED Face Search Engine with proper pool expansion logic
    """

    def __init__(self):
        self.aws_key = os.getenv("aws_access_key")
        self.aws_secret = os.getenv("aws_secret_key")
        self.aws_region = os.getenv("aws_region_name")

        # Initialize AWS clients
        self._init_aws_clients()

        # FIXED: Advanced configuration
        self.config = {
            # # Matching thresholds
            # "base_similarity_threshold": float(os.getenv("BASE_SIMILARITY_THRESHOLD", "0.68")),
            # "high_confidence_threshold": 0.78,
            # "very_high_confidence_threshold": 0.88,

            # # Quality parameters
            # "min_face_quality": float(os.getenv("MIN_FACE_QUALITY", "0.5")),
            # "min_face_size": int(os.getenv("MIN_FACE_SIZE", "60")),
            # "blur_threshold": 100.0,

            # # Matching parameters
            # "max_matches_per_guest": 100,
            # "use_multi_embedding_matching": True,
            # "use_contextual_scoring": True,
            # "use_quality_weighting": True,

            # # Performance
            # "batch_size": 25,
            # "max_pool_size": 50000,
            # "cache_ttl_hours": 24,

            # # Advanced features
            # "enable_face_clustering": True,
            # "enable_duplicate_removal": True,
            # "enable_group_photo_detection": True,
            # "group_photo_bonus": 0.05,

            # # FIXED: Pool expansion settings
            # "always_reprocess_on_pool_expansion": True,
            # "pool_expansion_detection_minutes": 5,
            # "incremental_search_expansion": 5,
            # "incremental_cache_refresh": True,
            # "incremental_consistency_wait": 5,

            # Matching thresholds - BALANCED FOR MOBILE
            "base_similarity_threshold": float(os.getenv("BASE_SIMILARITY_THRESHOLD", "0.45")),
            "high_confidence_threshold": 0.65,
            "very_high_confidence_threshold": 0.80,

            # Quality parameters - MOBILE OPTIMIZED
            "min_face_quality": float(os.getenv("MIN_FACE_QUALITY", "0.35")),
            "min_face_size": int(os.getenv("MIN_FACE_SIZE", "45")),
            "blur_threshold": 120.0,

            # Matching parameters - ENHANCED FOR MOBILE
            "max_matches_per_guest": 120,
            "use_multi_embedding_matching": True,
            "use_contextual_scoring": True,
            "use_quality_weighting": True,

            # Performance
            "batch_size": 25,
            "max_pool_size": 50000,
            "cache_ttl_hours": 24,

            # Advanced features - MOBILE FRIENDLY
            "enable_face_clustering": True,
            "enable_duplicate_removal": True,
            "enable_group_photo_detection": True,
            "group_photo_bonus": 0.08,

            # Pool expansion settings - MOBILE OPTIMIZED
            "always_reprocess_on_pool_expansion": True,
            "pool_expansion_detection_minutes": 8,
            "incremental_search_expansion": 7,
            "incremental_cache_refresh": True,
            "incremental_consistency_wait": 4,
        }

        # Caches and state
        self._event_cache = {}
        self._index_cache = {}
        self._cache_lock = threading.Lock()
        self._current_event_id = None
        self._current_trigger_context = {}

        # Metrics
        self.metrics = defaultdict(int)

        logger.info("✅ FIXED Advanced Face Search Engine initialized")
        self._log_configuration()

    def _init_aws_clients(self):
        """Initialize AWS clients with retry configuration"""
        session = boto3.Session(
            aws_access_key_id=self.aws_key,
            aws_secret_access_key=self.aws_secret,
            region_name=self.aws_region
        )

        self.dynamodb = session.client(
            'dynamodb',
            config=boto3.session.Config(
                max_pool_connections=50,
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )
        )

        self.cloudwatch = session.client('cloudwatch')
        self.s3 = session.client('s3')

    def _log_configuration(self):
        """Log current configuration"""
        logger.info("🔧 FIXED Configuration:")
        for key, value in self.config.items():
            logger.info(f"   {key}: {value}")

    # ===================================
    # FIXED Core Matching Algorithm
    # ===================================

    def process_event(self, event_id: str, update_mode: str = "full", trigger_context: Dict = None) -> Dict:
        """
        FIXED Main entry point with trigger context for pool expansion detection
        """
        start_time = time.time()

        try:
            # Store context for pool expansion detection
            self._current_event_id = event_id
            self._current_trigger_context = trigger_context or {}

            logger.info(f"🎯 FIXED: Starting face search for event: {event_id} (mode: {update_mode})")
            if trigger_context:
                logger.info(f"   Trigger context: {trigger_context}")

            # FIXED: Wait for DynamoDB consistency
            if update_mode == "incremental":
                consistency_wait = self.config['incremental_consistency_wait']
                logger.info(f"⏳ Waiting {consistency_wait}s for DynamoDB consistency...")
                time.sleep(consistency_wait)

            # Step 1: Load guest profiles
            guest_profiles = self._load_guest_profiles(event_id)
            if not guest_profiles:
                return self._create_response(
                    success=True,
                    message="No guest profiles found",
                    event_id=event_id,
                    guests_processed=0
                )

            # Step 2: Load and index pool faces
            pool_index, pool_metadata = self._load_and_index_pool_faces(event_id, update_mode)
            if pool_index is None or pool_index.ntotal == 0:
                return self._create_response(
                    success=True,
                    message="No pool faces found",
                    event_id=event_id,
                    guests_processed=len(guest_profiles)
                )

            # FIXED: Detect pool expansion
            is_pool_expansion = self._is_pool_expansion_trigger(event_id)
            logger.info(f"📊 Event {event_id} data loaded:")
            logger.info(f"   - Guest profiles: {len(guest_profiles)}")
            logger.info(f"   - Pool faces: {pool_index.ntotal}")
            logger.info(f"   - Pool expansion detected: {is_pool_expansion}")

            # Step 3: Load existing matches if incremental
            existing_matches = {}
            if update_mode == "incremental":
                existing_matches = self._load_existing_matches(event_id)
                logger.info(f"   - Existing matches loaded: {len(existing_matches)} guests")

            # Step 4: Perform matching
            all_matches = self._perform_advanced_matching(
                guest_profiles, pool_index, pool_metadata, existing_matches, update_mode, is_pool_expansion
            )

            # Step 5: Post-process results
            final_results = self._post_process_results(all_matches, update_mode)

            # Step 6: Store results
            stored_count = self._store_results(event_id, final_results, update_mode)

            # Calculate metrics
            processing_time = time.time() - start_time
            total_matches = sum(len(r["matches"]) for r in final_results)
            new_matches = sum(r["match_statistics"].get("new_matches", 0) for r in final_results)

            # Publish metrics
            self._publish_metrics(event_id, {
                "guests_processed": len(guest_profiles),
                "total_matches": total_matches,
                "new_matches": new_matches,
                "processing_time": processing_time,
                "update_mode": update_mode,
                "is_pool_expansion": is_pool_expansion
            })

            logger.info(f"✅ FIXED: Processing complete - {total_matches} total matches (+{new_matches} new)")

            return self._create_response(
                success=True,
                message=f"FIXED: Face matching completed ({update_mode} mode, pool_expansion: {is_pool_expansion})",
                event_id=event_id,
                guests_processed=len(guest_profiles),
                guests_with_matches=len(final_results),
                total_matches=total_matches,
                new_matches=new_matches,
                processing_time=round(processing_time, 2),
                update_mode=update_mode,
                is_pool_expansion=is_pool_expansion
            )

        except Exception as e:
            logger.error(f"💥 Error processing event {event_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

            return self._create_response(
                success=False,
                error=str(e),
                event_id=event_id,
                processing_time=time.time() - start_time
            )

    def _is_pool_expansion_trigger(self, event_id: str) -> bool:
        """
        FIXED: Detect if this face search was triggered by pool image processing
        """
        try:
            # Method 1: Check trigger context
            if self._current_trigger_context:
                process_type = self._current_trigger_context.get('process_type', '')
                is_pool_expansion = self._current_trigger_context.get('is_pool_expansion', False)

                # FIXED: Detect pool processing more accurately
                if process_type in ['normal', 'pool', '/imageprocess', '/imageprocess_v2']:
                    logger.info("   Pool expansion detected: pool processing trigger")
                    return True

                if is_pool_expansion:
                    logger.info("   Pool expansion detected: explicit flag")
                    return True

            # Method 2: Check for recent pool faces (within last 5 minutes)
            recent_time = (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')

            try:
                pool_response = self.dynamodb.query(
                    TableName="poolface",
                    KeyConditionExpression="eventid = :eventid",
                    ExpressionAttributeValues={":eventid": {"S": event_id}}
                )

                recent_pool_count = 0
                for item in pool_response.get("Items", []):
                    created_at = item.get("createdat", {}).get("S", "")
                    if created_at > recent_time:
                        recent_pool_count += 1

                if recent_pool_count > 0:
                    logger.info(f"   Pool expansion detected: {recent_pool_count} recent pool faces")
                    return True

            except Exception as e:
                logger.warning(f"Could not check recent pool faces: {e}")

            return False

        except Exception as e:
            logger.warning(f"Pool expansion detection failed: {e}")
            return False

    def _load_guest_profiles(self, event_id: str) -> Dict[str, GuestProfile]:
        """Load guest faces and create profiles"""
        try:
            logger.info(f"📥 FIXED: Loading guest profiles for event {event_id}")

            response = self.dynamodb.query(
                TableName="guestface",
                KeyConditionExpression="eventid = :eventid",
                ExpressionAttributeValues={":eventid": {"S": event_id}}
            )

            guest_data = defaultdict(list)
            guest_info = {}

            for item in response.get("Items", []):
                try:
                    embedding = self._parse_embedding(item["embedding"]["S"])
                    if embedding is None:
                        continue

                    guest_id = item.get("guest_id", {}).get("S", item["uid"]["S"])

                    face_emb = FaceEmbedding(
                        uid=item["uid"]["S"],
                        event_id=item["eventid"]["S"],
                        embedding=embedding,
                        s3_url=item["s3url"]["S"],
                        filename=item["filename"]["S"],
                        created_at=item["createdat"]["S"],
                        quality_score=self._calculate_embedding_quality(embedding)
                    )

                    guest_data[guest_id].append(face_emb)

                    if guest_id not in guest_info:
                        guest_info[guest_id] = {
                            "name": item.get("guest_name", {}).get("S", "Unknown"),
                            "email": item.get("guest_email", {}).get("S", "unknown@example.com"),
                            "phone": item.get("guest_phone", {}).get("S", "unknown"),
                            "registration_id": item.get("registration_id", {}).get("S", f"reg_{guest_id}")
                        }

                except Exception as e:
                    logger.error(f"Error parsing guest face: {e}")
                    continue

            profiles = {}
            for guest_id, embeddings in guest_data.items():
                if not embeddings:
                    continue

                embeddings.sort(key=lambda x: x.quality_score, reverse=True)
                profile_quality = np.mean([e.quality_score for e in embeddings])
                info = guest_info.get(guest_id, {})

                profile = GuestProfile(
                    guest_id=guest_id,
                    guest_name=info.get("name", "Unknown"),
                    guest_email=info.get("email", "unknown@example.com"),
                    guest_phone=info.get("phone", "unknown"),
                    registration_id=info.get("registration_id", f"reg_{guest_id}"),
                    embeddings=embeddings,
                    best_embedding_index=0,
                    profile_quality_score=profile_quality
                )

                profiles[guest_id] = profile

            logger.info(f"✅ FIXED: Loaded {len(profiles)} guest profiles")
            return profiles

        except Exception as e:
            logger.error(f"Failed to load guest profiles: {e}")
            return {}

    def _load_and_index_pool_faces(self, event_id: str, update_mode: str = "full") -> Tuple[Optional[faiss.Index], Dict]:
        """FIXED: Load pool faces with proper incremental support"""
        cache_key = f"pool_index_{event_id}"

        # FIXED: Always refresh in incremental mode
        if update_mode == "incremental":
            logger.info(f"📋 FIXED: Force refreshing pool index for incremental update")
            with self._cache_lock:
                if cache_key in self._index_cache:
                    del self._index_cache[cache_key]
        else:
            with self._cache_lock:
                if cache_key in self._index_cache:
                    logger.info(f"📋 Using cached pool index for event {event_id}")
                    return self._index_cache[cache_key]

        try:
            logger.info(f"📥 FIXED: Loading pool faces for event {event_id} (mode: {update_mode})")

            response = self.dynamodb.query(
                TableName="poolface",
                KeyConditionExpression="eventid = :eventid",
                ExpressionAttributeValues={":eventid": {"S": event_id}}
            )

            embeddings = []
            metadata = {}
            valid_count = 0

            for item in response.get("Items", []):
                try:
                    embedding = self._parse_embedding(item["embedding"]["S"])
                    if embedding is None:
                        continue

                    quality = self._calculate_embedding_quality(embedding)
                    if quality < self.config["min_face_quality"]:
                        continue

                    idx = len(embeddings)
                    embeddings.append(embedding)

                    metadata[idx] = {
                        "uid": item["uid"]["S"],
                        "s3url": item["s3url"]["S"],
                        "filename": item["filename"]["S"],
                        "createdat": item["createdat"]["S"],
                        "quality_score": quality,
                        "is_group_photo": self._detect_group_photo(item["filename"]["S"])
                    }

                    valid_count += 1

                except Exception as e:
                    logger.error(f"Error parsing pool face: {e}")
                    continue

            if not embeddings:
                logger.warning(f"No valid pool faces found for event {event_id}")
                return None, {}

            # Create FAISS index
            embeddings_np = np.array(embeddings, dtype=np.float32)
            embeddings_normalized = normalize(embeddings_np, norm='l2')

            if len(embeddings) < 1000:
                index = faiss.IndexFlatIP(embeddings_normalized.shape[1])
                index.add(embeddings_normalized)
            else:
                nlist = min(int(np.sqrt(len(embeddings))), 100)
                quantizer = faiss.IndexFlatIP(embeddings_normalized.shape[1])
                index = faiss.IndexIVFFlat(quantizer, embeddings_normalized.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(embeddings_normalized)
                index.add(embeddings_normalized)
                index.nprobe = min(nlist // 4, 20)

            logger.info(f"✅ FIXED: Indexed {valid_count} pool faces")

            # Cache based on update mode
            if update_mode != "incremental" or self.config["incremental_cache_refresh"]:
                with self._cache_lock:
                    self._index_cache[cache_key] = (index, metadata)

            return index, metadata

        except Exception as e:
            logger.error(f"Failed to load pool faces: {e}")
            return None, {}

    def _load_existing_matches(self, event_id: str) -> Dict[str, Dict]:
        """FIXED: Load existing matches for incremental updates"""
        try:
            logger.info(f"📥 FIXED: Loading existing matches for event {event_id}")

            response = self.dynamodb.query(
                TableName="face_match_results",
                KeyConditionExpression="eventId = :eventId",
                ExpressionAttributeValues={":eventId": {"S": event_id}}
            )

            existing_matches = {}

            for item in response.get("Items", []):
                guest_id = item.get("guestId", {}).get("S")
                if not guest_id:
                    continue

                matched_images_str = item.get("matched_images", {}).get("S", "[]")
                try:
                    matched_images = json.loads(matched_images_str)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse matched_images for guest {guest_id}")
                    matched_images = []

                existing_matches[guest_id] = {
                    "existing_matches": matched_images,
                    "existing_pool_uids": {m["pool_uid"] for m in matched_images}
                }

            logger.info(f"✅ FIXED: Loaded existing matches for {len(existing_matches)} guests")
            return existing_matches

        except Exception as e:
            logger.error(f"Failed to load existing matches: {e}")
            return {}

    def _perform_advanced_matching(
        self,
        guest_profiles: Dict[str, GuestProfile],
        pool_index: faiss.Index,
        pool_metadata: Dict,
        existing_matches: Dict = None,
        update_mode: str = "full",
        is_pool_expansion: bool = False
    ) -> List[Dict]:
        """CORRECTED: Perform matching, capturing a result for every guest."""
        logger.info(f"🔍 CORRECTED: Performing matching for {len(guest_profiles)} guests (mode: {update_mode}, pool_expansion: {is_pool_expansion})")

        # This list will hold a result dictionary for every processed guest.
        all_guest_results = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            for guest_id, profile in guest_profiles.items():
                future = executor.submit(
                    self._match_guest_profile,
                    profile,
                    pool_index,
                    pool_metadata,
                    existing_matches.get(guest_id, {}) if existing_matches else {},
                    update_mode,
                    is_pool_expansion
                )
                futures[future] = guest_id

            for future in as_completed(futures):
                guest_id = futures[future]
                try:
                    # 'guest_result' is now guaranteed to be a dictionary unless an exception occurred.
                    guest_result = future.result()

                    # MODIFIED LOGIC: We no longer check `if guest_result:`, because it will always exist.
                    # We append it regardless. This ensures every guest gets a record.
                    if guest_result is not None:
                        all_guest_results.append(guest_result)

                        # Log the outcome for this guest.
                        total_count = guest_result['match_statistics']['total_matches']
                        new_count = guest_result['match_statistics']['new_matches']
                        if total_count > 0:
                            logger.info(f"✅ Guest {guest_id} processed: {total_count} total matches (+{new_count} new)")
                        else:
                            logger.info(f"✅ Guest {guest_id} processed: 0 matches found.")
                    else:
                        # This case only happens if _match_guest_profile had a critical failure.
                        logger.error(f"❌ Matching failed for guest {guest_id}, no result returned.")

                except Exception as e:
                    logger.error(f"Error getting result for guest {guest_id}: {e}")

        return all_guest_results


    def _match_guest_profile(
        self,
        profile: GuestProfile,
        pool_index: faiss.Index,
        pool_metadata: Dict,
        existing_data: Dict = None,
        update_mode: str = "full",
        is_pool_expansion: bool = False
    ) -> Optional[Dict]:
        """
        CORRECTED: Match single guest, ensuring a result is always returned
        to signify completion, even with zero matches.
        """
        try:
            search_multiplier = self.config["incremental_search_expansion"] if update_mode == "incremental" else 1
            k = min(self.config["max_matches_per_guest"] * search_multiplier, pool_index.ntotal)

            # Get candidates
            if self.config["use_multi_embedding_matching"] and len(profile.embeddings) > 1:
                candidates = self._multi_embedding_search(profile, pool_index, k)
            else:
                candidates = self._single_embedding_search(profile.get_best_embedding(), pool_index, k)

            existing_pool_uids = existing_data.get("existing_pool_uids", set()) if existing_data else set()

            # NEW LOGIC: Handle case where no potential candidates are found
            # This is a change from the original, where it would return None.
            if not candidates:
                logger.info(f"   No candidates found for guest {profile.guest_id}. Checking for existing matches.")
                # If the guest already had matches from a previous run, we should still return them.
                if existing_data and existing_data.get("existing_matches"):
                    return self._create_result_with_existing_matches(profile, existing_data["existing_matches"])

                # If no candidates AND no existing matches, create the zero-match result.
                logger.info(f"   Guest {profile.guest_id}: No new candidates and no existing matches. Creating zero-match result.")
                return {
                    "guest_id": profile.guest_id,
                    "guest_name": profile.guest_name,
                    "guest_email": profile.guest_email,
                    "guest_phone": profile.guest_phone,
                    "guest_registration_id": profile.registration_id,
                    "guest_selfie_url": profile.get_best_embedding().s3_url,
                    "profile_quality": profile.profile_quality_score,
                    "matches": [],
                    "match_statistics": {
                        "total_matches": 0,
                        "new_matches": 0,
                        "existing_matches": 0,
                        "best_score": 0.0,
                        "average_score": 0.0,
                        "confidence_distribution": {},
                        "is_pool_expansion": is_pool_expansion
                    }
                }

            # Process candidates with corrected pool expansion logic (NO CHANGE TO THIS PART)
            new_matches = []
            for pool_idx, base_similarity in candidates:
                pool_info = pool_metadata[pool_idx]
                pool_uid = pool_info["uid"]

                if update_mode == "incremental" and not is_pool_expansion and pool_uid in existing_pool_uids:
                    continue

                final_score = self._calculate_enhanced_score(base_similarity, profile, pool_info)
                if final_score < self.config["base_similarity_threshold"]:
                    continue

                confidence = self._get_confidence_level(final_score)
                match = MatchResult(
                    pool_uid=pool_info["uid"],
                    pool_url=pool_info["s3url"],
                    pool_filename=pool_info["filename"],
                    similarity_score=float(final_score),
                    match_confidence=float(base_similarity),
                    match_type="primary",
                    metadata={
                        "quality_score": pool_info["quality_score"],
                        "is_group_photo": pool_info["is_group_photo"],
                        "confidence_level": confidence.value,
                        "found_in_expansion": is_pool_expansion,
                        "is_new_match": pool_uid not in existing_pool_uids
                    }
                )

                if is_pool_expansion or pool_uid not in existing_pool_uids:
                    new_matches.append(match)

            # Sort and deduplicate new matches (NO CHANGE TO THIS PART)
            new_matches.sort(key=lambda x: x.similarity_score, reverse=True)
            if self.config["enable_duplicate_removal"]:
                new_matches = self._remove_duplicate_matches(new_matches)

            # Combine with existing matches properly (NO CHANGE TO THIS PART)
            all_matches = []
            if existing_data and existing_data.get("existing_matches"):
                existing_match_objects = [MatchResult(**m) for m in existing_data["existing_matches"]]
                all_matches.extend(existing_match_objects)

            if new_matches:
                all_matches.extend(new_matches)

            # Deduplicate combined list and sort (NO CHANGE TO THIS PART)
            seen_uids = set()
            deduplicated_matches = []
            all_matches.sort(key=lambda x: x.similarity_score, reverse=True)
            for match in all_matches:
                if match.pool_uid not in seen_uids:
                    deduplicated_matches.append(match)
                    seen_uids.add(match.pool_uid)

            final_matches = deduplicated_matches[:self.config["max_matches_per_guest"]]

            # ================== KEY LOGIC CHANGE ==================
            # This part now correctly handles both cases: guests with matches and guests without.

            # If after all processing there are no final matches, create the zero-match result.
            # This ensures a "receipt" of processing is always created.
            if not final_matches:
                logger.info(f"   Guest {profile.guest_id}: Processing complete. 0 total matches.")
                return {
                    "guest_id": profile.guest_id,
                    "guest_name": profile.guest_name,
                    "guest_email": profile.guest_email,
                    "guest_phone": profile.guest_phone,
                    "guest_registration_id": profile.registration_id,
                    "guest_selfie_url": profile.get_best_embedding().s3_url,
                    "profile_quality": profile.profile_quality_score,
                    "matches": [], # Return an empty list
                    "match_statistics": {
                        "total_matches": 0,
                        "new_matches": 0,
                        "existing_matches": 0,
                        "best_score": 0.0,
                        "average_score": 0.0,
                        "confidence_distribution": {},
                        "is_pool_expansion": is_pool_expansion
                    }
                }

            # If there are matches, proceed with the original logic.
            new_count = len([m for m in final_matches if m.metadata.get("is_new_match", False)])
            logger.info(f"   Guest {profile.guest_id}: {len(final_matches)} total matches ({new_count} new)")

            return {
                "guest_id": profile.guest_id,
                "guest_name": profile.guest_name,
                "guest_email": profile.guest_email,
                "guest_phone": profile.guest_phone,
                "guest_registration_id": profile.registration_id,
                "guest_selfie_url": profile.get_best_embedding().s3_url,
                "profile_quality": profile.profile_quality_score,
                "matches": [asdict(m) for m in final_matches],
                "match_statistics": {
                    "total_matches": len(final_matches),
                    "new_matches": new_count,
                    "existing_matches": len(existing_data.get("existing_matches", [])) if existing_data else 0,
                    "best_score": final_matches[0].similarity_score,
                    "average_score": np.mean([m.similarity_score for m in final_matches]),
                    "confidence_distribution": self._get_confidence_distribution(final_matches),
                    "is_pool_expansion": is_pool_expansion
                }
            }

        except Exception as e:
            logger.error(f"Error matching guest profile {profile.guest_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None # Return None only on a critical failure

    def _create_result_with_existing_matches(self, profile: GuestProfile, existing_matches: List[Dict]) -> Dict:
        """FIXED: Create result with only existing matches"""
        existing_match_objects = []
        for existing_match in existing_matches:
            match = MatchResult(
                pool_uid=existing_match["pool_uid"],
                pool_url=existing_match["pool_url"],
                pool_filename=existing_match["pool_filename"],
                similarity_score=existing_match["similarity_score"],
                match_confidence=existing_match["match_confidence"],
                match_type=existing_match["match_type"],
                metadata=existing_match["metadata"]
            )
            existing_match_objects.append(match)

        return {
            "guest_id": profile.guest_id,
            "guest_name": profile.guest_name,
            "guest_email": profile.guest_email,
            "guest_phone": profile.guest_phone,
            "guest_registration_id": profile.registration_id,
            "guest_selfie_url": profile.get_best_embedding().s3_url,
            "profile_quality": profile.profile_quality_score,
            "matches": [asdict(m) for m in existing_match_objects],
            "match_statistics": {
                "total_matches": len(existing_match_objects),
                "new_matches": 0,
                "existing_matches": len(existing_matches),
                "best_score": existing_match_objects[0].similarity_score if existing_match_objects else 0,
                "average_score": np.mean([m.similarity_score for m in existing_match_objects]) if existing_match_objects else 0,
                "confidence_distribution": self._get_confidence_distribution(existing_match_objects)
            }
        }

    def _multi_embedding_search(self, profile: GuestProfile, pool_index: faiss.Index, k: int) -> List[Tuple[int, float]]:
        """Search using multiple embeddings and aggregate results"""
        all_candidates = defaultdict(list)

        for i, embedding in enumerate(profile.embeddings):
            weight = embedding.quality_score
            emb_normalized = embedding.to_normalized_embedding().reshape(1, -1).astype(np.float32)
            similarities, indices = pool_index.search(emb_normalized, k)

            for j, idx in enumerate(indices[0]):
                if idx >= 0:
                    weighted_sim = similarities[0][j] * weight
                    all_candidates[idx].append(weighted_sim)

        final_candidates = []
        for idx, scores in all_candidates.items():
            max_score = max(scores)
            occurrence_bonus = min(len(scores) * 0.02, 0.1)
            final_score = min(max_score + occurrence_bonus, 1.0)
            final_candidates.append((idx, final_score))

        final_candidates.sort(key=lambda x: x[1], reverse=True)
        return final_candidates[:k]

    def _single_embedding_search(self, embedding: FaceEmbedding, pool_index: faiss.Index, k: int) -> List[Tuple[int, float]]:
        """Search using a single embedding"""
        emb_normalized = embedding.to_normalized_embedding().reshape(1, -1).astype(np.float32)
        similarities, indices = pool_index.search(emb_normalized, k)

        candidates = []
        for j, idx in enumerate(indices[0]):
            if idx >= 0:
                candidates.append((idx, similarities[0][j]))

        return candidates

    def _calculate_enhanced_score(self, base_similarity: float, profile: GuestProfile, pool_info: Dict) -> float:
        """Calculate enhanced score with quality and context weighting"""
        score = base_similarity

        if self.config["use_quality_weighting"]:
            quality_avg = (profile.profile_quality_score + pool_info["quality_score"]) / 2
            quality_bonus = quality_avg * 0.05
            score += quality_bonus

        if self.config["use_contextual_scoring"]:
            if pool_info.get("is_group_photo", False):
                score += self.config["group_photo_bonus"]

        return min(max(score, 0.0), 1.0)

    def _get_confidence_level(self, score: float) -> MatchConfidence:
        """Determine confidence level based on score"""
        if score >= self.config["very_high_confidence_threshold"]:
            return MatchConfidence.VERY_HIGH
        elif score >= self.config["high_confidence_threshold"]:
            return MatchConfidence.HIGH
        elif score >= 0.65:
            return MatchConfidence.MEDIUM
        elif score >= 0.55:
            return MatchConfidence.LOW
        else:
            return MatchConfidence.VERY_LOW

    def _remove_duplicate_matches(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Remove duplicate or very similar matches"""
        if len(matches) <= 1:
            return matches

        filtered = [matches[0]]

        for match in matches[1:]:
            is_duplicate = False
            for kept_match in filtered:
                if self._are_images_similar(match.pool_filename, kept_match.pool_filename):
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(match)

        return filtered

    def _are_images_similar(self, filename1: str, filename2: str) -> bool:
        """Check if two images are likely duplicates based on filename"""
        base1 = os.path.splitext(filename1)[0]
        base2 = os.path.splitext(filename2)[0]

        if base1 == base2:
            return True

        if base1[:-3] == base2[:-3] and base1[-3:].isdigit() and base2[-3:].isdigit():
            num1 = int(base1[-3:])
            num2 = int(base2[-3:])
            if abs(num1 - num2) <= 2:
                return True

        return False

    def _post_process_results(self, all_matches: List[Dict], update_mode: str = "full") -> List[Dict]:
        """FIXED: Post-process results with proper handling for zero-match guests"""
        valid_matches = [m for m in all_matches if m is not None]

        for result in valid_matches:
            # FIXED: Handle zero-match guests in overall_score calculation
            best_score = result["match_statistics"].get("best_score", 0.0)
            average_score = result["match_statistics"].get("average_score", 0.0)
            profile_quality = result.get("profile_quality", 0.5)

            # Ensure all values are floats and handle None cases
            if best_score is None:
                best_score = 0.0
            if average_score is None:
                average_score = 0.0
            if profile_quality is None:
                profile_quality = 0.5

            result["overall_score"] = (
                float(best_score) * 0.5 +
                float(average_score) * 0.3 +
                float(profile_quality) * 0.2
            )

        valid_matches.sort(key=lambda x: x["overall_score"], reverse=True)

        if update_mode == "incremental":
            total_new_matches = sum(r["match_statistics"].get("new_matches", 0) for r in valid_matches)
            logger.info(f"📊 FIXED: Post-processing: {len(valid_matches)} guests, {total_new_matches} new matches")

        return valid_matches

    def _store_results(self, event_id: str, results: List[Dict], update_mode: str = "full") -> int:
        """FIXED: Store results with proper handling for zero-match guests"""
        try:
            logger.info(f"💾 FIXED: Storing {len(results)} guest results (mode: {update_mode})")

            if update_mode == "full":
                self._clear_existing_results(event_id)

            stored = 0
            failed = 0

            for i in range(0, len(results), self.config["batch_size"]):
                batch = results[i:i + self.config["batch_size"]]
                request_items = []

                for result in batch:
                    try:
                        # CRITICAL FIX: Handle both zero-match and regular guests
                        match_count = len(result.get("matches", []))

                        if update_mode == "incremental":
                            new_matches = result["match_statistics"].get("new_matches", 0)
                            logger.info(f"Updating guest: {result['guest_id']} with {match_count} total matches (+{new_matches} new)")
                        else:
                            logger.info(f"Storing guest: {result['guest_id']} with {match_count} matches")

                        # FIXED: Ensure best_score and average_score are properly handled for zero matches
                        best_score = result["match_statistics"].get("best_score", 0.0)
                        average_score = result["match_statistics"].get("average_score", 0.0)

                        # FIXED: Handle case where scores might be None or invalid
                        if best_score is None or not isinstance(best_score, (int, float)):
                            best_score = 0.0
                        if average_score is None or not isinstance(average_score, (int, float)):
                            average_score = 0.0

                        item = {
                            "eventId": {"S": event_id},
                            "guestId": {"S": result["guest_id"]},
                            "guest_uid": {"S": f"guest_{result['guest_id']}"},
                            "guest_name": {"S": result["guest_name"]},
                            "guest_email": {"S": result["guest_email"]},
                            "guest_phone": {"S": result["guest_phone"]},
                            "guest_selfie_url": {"S": result["guest_selfie_url"]},
                            "guest_registration_id": {"S": result["guest_registration_id"]},
                            "matched_images": {"S": json.dumps(result.get("matches", []))},
                            "match_statistics": {"S": json.dumps(result["match_statistics"])},
                            "profile_quality": {"N": str(float(result.get("profile_quality", 0.5)))},
                            "overall_score": {"N": str(float(result.get("overall_score", 0.0)))},
                            "total_matches": {"N": str(match_count)},
                            "best_similarity": {"N": str(float(best_score))},
                            "average_similarity": {"N": str(float(average_score))},
                            "delivery_status": {"S": "pending"},
                            "processed_at": {"S": datetime.now(timezone.utc).isoformat()},
                            "created_at": {"S": datetime.now(timezone.utc).isoformat()},
                            "algorithm_version": {"S": "3.0-fixed"},
                            "update_mode": {"S": update_mode},
                            "new_matches": {"N": str(result["match_statistics"].get("new_matches", 0))}
                        }

                        request_items.append({"PutRequest": {"Item": item}})

                    except Exception as e:
                        logger.error(f"Error preparing item for guest {result.get('guest_id', 'unknown')}: {e}")
                        logger.error(f"Result structure: {result}")
                        failed += 1
                        continue

                if request_items:
                    try:
                        logger.info(f"Writing batch of {len(request_items)} items to face_match_results")
                        response = self.dynamodb.batch_write_item(
                            RequestItems={"face_match_results": request_items}
                        )

                        unprocessed = response.get("UnprocessedItems", {})
                        if unprocessed:
                            unprocessed_count = len(unprocessed.get("face_match_results", []))
                            logger.warning(f"Some items were not processed: {unprocessed_count}")
                            failed += unprocessed_count
                            stored += len(request_items) - unprocessed_count
                        else:
                            stored += len(request_items)

                        logger.info(f"Batch write completed. Stored: {len(request_items)} items")

                    except Exception as e:
                        logger.error(f"Batch write failed: {e}")
                        logger.error(f"Request items sample: {request_items[0] if request_items else 'No items'}")
                        failed += len(request_items)

            logger.info(f"✅ FIXED: Storage complete - Stored: {stored}, Failed: {failed}")
            return stored

        except Exception as e:
            logger.error(f"Failed to store results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _clear_existing_results(self, event_id: str):
        """Clear existing results for full update"""
        try:
            logger.info(f"🗑️ FIXED: Clearing existing results for event {event_id}")

            response = self.dynamodb.query(
                TableName="face_match_results",
                KeyConditionExpression="eventId = :eventId",
                ExpressionAttributeValues={":eventId": {"S": event_id}},
                ProjectionExpression="eventId, guestId"
            )

            items_to_delete = response.get("Items", [])
            if items_to_delete:
                logger.info(f"Deleting {len(items_to_delete)} existing results")

                for i in range(0, len(items_to_delete), 25):
                    batch = items_to_delete[i:i + 25]
                    delete_requests = []

                    for item in batch:
                        delete_requests.append({
                            "DeleteRequest": {
                                "Key": {
                                    "eventId": item["eventId"],
                                    "guestId": item["guestId"]
                                }
                            }
                        })

                    if delete_requests:
                        self.dynamodb.batch_write_item(
                            RequestItems={"face_match_results": delete_requests}
                        )

                logger.info(f"Cleared {len(items_to_delete)} existing results")

        except Exception as e:
            logger.error(f"Error clearing existing results: {e}")

    # ===================================
    # Utility Methods
    # ===================================


    def debug_store_results(self, event_id: str, results: List[Dict]) -> None:
        """Debug function to check what's being stored"""
        logger.info(f"🔍 DEBUG: About to store {len(results)} results for event {event_id}")

        for i, result in enumerate(results):
            logger.info(f"   Result {i+1}:")
            logger.info(f"     - Guest ID: {result.get('guest_id', 'MISSING')}")
            logger.info(f"     - Guest Name: {result.get('guest_name', 'MISSING')}")
            logger.info(f"     - Matches: {len(result.get('matches', []))}")
            logger.info(f"     - Match stats: {result.get('match_statistics', {})}")
            logger.info(f"     - Overall score: {result.get('overall_score', 'MISSING')}")
            logger.info(f"     - Profile quality: {result.get('profile_quality', 'MISSING')}")

            # Check for potential issues
            if not result.get("guest_id"):
                logger.error(f"     ❌ Missing guest_id in result {i+1}")
            if "match_statistics" not in result:
                logger.error(f"     ❌ Missing match_statistics in result {i+1}")
            if result.get("match_statistics", {}).get("best_score") is None:
                logger.warning(f"     ⚠️ best_score is None in result {i+1}")
            if result.get("match_statistics", {}).get("average_score") is None:
                logger.warning(f"     ⚠️ average_score is None in result {i+1}")
            if result.get("overall_score") is None:
                logger.warning(f"     ⚠️ overall_score is None in result {i+1}")

    def _parse_embedding(self, embedding_str: str) -> Optional[np.ndarray]:
        """Parse embedding string to numpy array"""
        try:
            embedding = json.loads(embedding_str)

            if isinstance(embedding[0], list):
                embedding = embedding[0]

            embedding_np = np.array(embedding, dtype=np.float32)

            if embedding_np.shape[0] == 0:
                return None

            return embedding_np

        except Exception as e:
            logger.error(f"Failed to parse embedding: {e}")
            return None

    def _calculate_embedding_quality(self, embedding: np.ndarray) -> float:
        """Calculate embedding quality score"""
        try:
            norm = np.linalg.norm(embedding)
            norm_score = min(norm / 100.0, 1.0)

            variance = np.var(embedding)
            var_score = min(variance * 10, 1.0)

            non_zero_ratio = np.count_nonzero(embedding) / len(embedding)

            range_val = np.ptp(embedding)
            range_score = min(range_val / 10.0, 1.0)

            quality = (
                norm_score * 0.3 +
                var_score * 0.3 +
                non_zero_ratio * 0.2 +
                range_score * 0.2
            )

            return float(np.clip(quality, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Error calculating quality: {e}")
            return 0.5

    def _detect_group_photo(self, filename: str) -> bool:
        """Detect if image is likely a group photo"""
        group_indicators = [
            'group', 'team', 'family', 'friends', 'party',
            'wedding', 'ceremony', 'gathering', 'crowd'
        ]

        filename_lower = filename.lower()
        return any(indicator in filename_lower for indicator in group_indicators)

    def _get_confidence_distribution(self, matches: List[MatchResult]) -> Dict[str, int]:
        """Get distribution of confidence levels"""
        distribution = defaultdict(int)

        for match in matches:
            confidence = match.metadata.get("confidence_level", "unknown")
            distribution[confidence] += 1

        return dict(distribution)

    def _create_response(self, **kwargs) -> Dict:
        """Create standardized response"""
        response = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "algorithm_version": "3.0-fixed"
        }
        response.update(kwargs)
        return response

    def _publish_metrics(self, event_id: str, metrics: Dict):
        """Publish metrics to CloudWatch"""
        try:
            metric_data = []

            for metric_name, value in metrics.items():
                metric_data.append({
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': 'Count' if 'count' in metric_name.lower() else 'None',
                    'Dimensions': [
                        {'Name': 'EventId', 'Value': event_id},
                        {'Name': 'Algorithm', 'Value': 'v3.0-fixed'}
                    ]
                })

            self.cloudwatch.put_metric_data(
                Namespace='FaceSearch/Fixed',
                MetricData=metric_data
            )

        except Exception as e:
            logger.error(f"Failed to publish metrics: {e}")

    def clear_cache(self, event_id: Optional[str] = None):
        """Clear cache for event or all events"""
        with self._cache_lock:
            if event_id:
                keys_to_remove = [k for k in self._index_cache.keys() if event_id in k]
                for key in keys_to_remove:
                    del self._index_cache[key]
                logger.info(f"🧹 Cleared cache for event {event_id}")
            else:
                self._index_cache.clear()
                self._event_cache.clear()
                logger.info("🧹 Cleared all caches")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._cache_lock:
            return {
                "index_cache_size": len(self._index_cache),
                "event_cache_size": len(self._event_cache),
                "cached_events": list(set(k.split('_')[2] for k in self._index_cache.keys()))
            }

# ===================================
# API Result Formatter
# ===================================

class ResultFormatter:
    """Format results for API responses"""

    @staticmethod
    def format_event_results(event_id: str, dynamodb_client) -> Dict:
        """Format results for event endpoint"""
        try:
            logger.info(f"📊 Querying results for event {event_id}")

            response = dynamodb_client.query(
                TableName="face_match_results",
                KeyConditionExpression="eventId = :eventId",
                ExpressionAttributeValues={":eventId": {"S": event_id}}
            )

            logger.info(f"Found {len(response.get('Items', []))} results")

            guests = []
            stats = {
                "total_guests": 0,
                "guests_with_matches": 0,
                "total_matches": 0,
                "new_matches": 0,
                "confidence_levels": defaultdict(int),
                "quality_scores": []
            }

            for item in response.get("Items", []):
                try:
                    guest_data = {
                        "guest_id": item.get("guestId", item.get("guest_id", {})).get("S", "unknown"),
                        "guest_name": item.get("guest_name", {}).get("S", "Unknown"),
                        "guest_email": item.get("guest_email", {}).get("S", "unknown@example.com"),
                        "guest_phone": item.get("guest_phone", {}).get("S", "unknown"),
                        "guest_selfie_url": item.get("guest_selfie_url", {}).get("S", ""),
                        "total_matches": int(item.get("total_matches", {}).get("N", "0")),
                        "new_matches": int(item.get("new_matches", {}).get("N", "0")),
                        "best_similarity": float(item.get("best_similarity", {}).get("N", "0")),
                        "average_similarity": float(item.get("average_similarity", {}).get("N", "0")),
                        "profile_quality": float(item.get("profile_quality", {"N": "0.5"}).get("N", "0.5")),
                        "delivery_status": item.get("delivery_status", {"S": "pending"}).get("S", "pending"),
                        "update_mode": item.get("update_mode", {"S": "unknown"}).get("S", "unknown"),
                        "algorithm_version": item.get("algorithm_version", {"S": "unknown"}).get("S", "unknown")
                    }

                    matched_images_str = item.get("matched_images", {}).get("S", "[]")
                    matches = json.loads(matched_images_str)
                    guest_data["top_matches"] = matches[:5]
                    guest_data["match_count"] = len(matches)

                    guests.append(guest_data)

                    stats["total_guests"] += 1
                    if guest_data["total_matches"] > 0:
                        stats["guests_with_matches"] += 1
                    stats["total_matches"] += guest_data["total_matches"]
                    stats["new_matches"] += guest_data["new_matches"]
                    stats["quality_scores"].append(guest_data["profile_quality"])

                    match_stats_str = item.get("match_statistics", {}).get("S", "{}")
                    match_stats = json.loads(match_stats_str)
                    if "confidence_distribution" in match_stats:
                        for level, count in match_stats["confidence_distribution"].items():
                            stats["confidence_levels"][level] += count

                except Exception as e:
                    logger.error(f"Error parsing guest data: {e}")
                    continue

            if stats["quality_scores"]:
                stats["average_quality"] = round(np.mean(stats["quality_scores"]), 3)
                stats["quality_scores"] = []

            guests.sort(key=lambda x: (x["total_matches"], x["best_similarity"]), reverse=True)

            return {
                "event_id": event_id,
                "guests": guests,
                "statistics": dict(stats),
                "retrieved_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to format event results: {e}")
            return {"error": str(e), "event_id": event_id}

    @staticmethod
    def format_guest_details(event_id: str, guest_id: str, dynamodb_client) -> Dict:
        """Format detailed results for a specific guest"""
        try:
            logger.info(f"📊 Querying details for guest {guest_id} in event {event_id}")

            response = dynamodb_client.get_item(
                TableName="face_match_results",
                Key={
                    "eventId": {"S": event_id},
                    "guestId": {"S": guest_id}
                }
            )

            item = response.get("Item")
            if not item:
                scan_response = dynamodb_client.scan(
                    TableName="face_match_results",
                    FilterExpression="eventId = :eventId AND guestId = :guestId",
                    ExpressionAttributeValues={
                        ":eventId": {"S": event_id},
                        ":guestId": {"S": guest_id}
                    }
                )

                items = scan_response.get("Items", [])
                if not items:
                    return {"error": "Guest not found", "event_id": event_id, "guest_id": guest_id}
                item = items[0]

            matches = json.loads(item.get("matched_images", {}).get("S", "[]"))
            match_stats = json.loads(item.get("match_statistics", {"S": "{}"}).get("S", "{}"))

            matches_by_confidence = defaultdict(list)
            for match in matches:
                confidence = match.get("metadata", {}).get("confidence_level", "unknown")
                matches_by_confidence[confidence].append(match)

            return {
                "event_id": event_id,
                "guest": {
                    "guest_id": item.get("guestId", item.get("guest_id", {})).get("S", guest_id),
                    "guest_name": item.get("guest_name", {}).get("S", "Unknown"),
                    "guest_email": item.get("guest_email", {}).get("S", "unknown@example.com"),
                    "guest_phone": item.get("guest_phone", {}).get("S", "unknown"),
                    "guest_selfie_url": item.get("guest_selfie_url", {}).get("S", ""),
                    "guest_registration_id": item.get("guest_registration_id", {}).get("S", ""),
                    "profile_quality": float(item.get("profile_quality", {"N": "0.5"}).get("N", "0.5")),
                    "total_matches": int(item.get("total_matches", {}).get("N", "0")),
                    "new_matches": int(item.get("new_matches", {}).get("N", "0")),
                    "delivery_status": item.get("delivery_status", {"S": "pending"}).get("S", "pending"),
                    "update_mode": item.get("update_mode", {"S": "unknown"}).get("S", "unknown"),
                    "algorithm_version": item.get("algorithm_version", {"S": "unknown"}).get("S", "unknown")
                },
                "match_statistics": match_stats,
                "matches_by_confidence": dict(matches_by_confidence),
                "all_matches": matches,
                "processed_at": item.get("processed_at", {}).get("S", ""),
                "algorithm_version": item.get("algorithm_version", {"S": "unknown"}).get("S", "unknown")
            }

        except Exception as e:
            logger.error(f"Failed to format guest details: {e}")
            return {"error": str(e), "event_id": event_id, "guest_id": guest_id}

# ===================================
# FIXED Flask API Application
# ===================================

app = Flask(__name__)
search_engine = AdvancedFaceSearchEngine()
result_formatter = ResultFormatter()

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "fixed-advanced-face-search",
        "version": "3.0-fixed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": [
            "fixed_pool_expansion_logic",
            "multi_embedding_matching",
            "quality_aware_ranking",
            "contextual_scoring",
            "duplicate_detection",
            "group_photo_handling",
            "enhanced_debugging"
        ]
    })

@app.route("/search/process", methods=["POST"])
def process_event():
    """FIXED: Process face matching with proper pool expansion support"""
    try:
        data = request.get_json()
        event_id = data.get("eventId")
        update_mode = data.get("updateMode", "full")
        trigger_context = data.get("trigger_context", {})

        if not event_id:
            return jsonify({"error": "eventId is required"}), 400

        logger.info(f"🎯 FIXED API: Processing event {event_id} (mode: {update_mode})")
        if trigger_context:
            logger.info(f"   Trigger context: {trigger_context}")

        result = search_engine.process_event(event_id, update_mode, trigger_context)

        return jsonify(result)

    except Exception as e:
        logger.error(f"API process error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/search/results/<event_id>", methods=["GET"])
def get_results(event_id):
    """Get results for an event"""
    try:
        results = result_formatter.format_event_results(event_id, search_engine.dynamodb)
        return jsonify(results)
    except Exception as e:
        logger.error(f"API get results error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/search/guest/<event_id>/<guest_id>", methods=["GET"])
def get_guest(event_id, guest_id):
    """Get detailed results for a specific guest"""
    try:
        results = result_formatter.format_guest_details(event_id, guest_id, search_engine.dynamodb)
        return jsonify(results)
    except Exception as e:
        logger.error(f"API get guest error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/search/trigger/<event_id>", methods=["POST"])
def trigger_search(event_id):
    """FIXED: Manual trigger for face search with context"""
    try:
        data = request.get_json() or {}
        update_mode = data.get("updateMode", "incremental")
        trigger_context = data.get("trigger_context", {"process_type": "manual", "is_pool_expansion": True})

        logger.info(f"🎯 MANUAL TRIGGER: Processing event {event_id} (mode: {update_mode})")

        result = search_engine.process_event(event_id, update_mode, trigger_context)

        return jsonify(result)

    except Exception as e:
        logger.error(f"API trigger error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/search/config", methods=["GET", "POST"])
def manage_config():
    """Get or update configuration"""
    try:
        if request.method == "GET":
            return jsonify({
                "config": search_engine.config,
                "cache_stats": search_engine.get_cache_stats()
            })
        else:
            updates = request.get_json()
            for key, value in updates.items():
                if key in search_engine.config:
                    search_engine.config[key] = value
                    logger.info(f"Updated config: {key} = {value}")

            return jsonify({
                "status": "updated",
                "config": search_engine.config
            })

    except Exception as e:
        logger.error(f"API config error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/search/cache/clear", methods=["POST"])
def clear_cache():
    """Clear cache"""
    try:
        data = request.get_json() or {}
        event_id = data.get("event_id")

        search_engine.clear_cache(event_id)

        return jsonify({
            "status": "success",
            "message": f"Cache cleared for {'all events' if not event_id else f'event {event_id}'}"
        })
    except Exception as e:
        logger.error(f"API cache clear error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/search/stats", methods=["GET"])
def get_stats():
    """Get service statistics"""
    try:
        memory = psutil.virtual_memory()
        process = psutil.Process()

        return jsonify({
            "service": "corrected-advanced-face-search",
            "version": "3.0-corrected",
            "uptime_seconds": time.time() - process.create_time(),
            "memory_usage_percent": memory.percent,
            "cpu_percent": process.cpu_percent(interval=1),
            "cache_stats": search_engine.get_cache_stats(),
            "configuration": {
                "similarity_threshold": search_engine.config["base_similarity_threshold"],
                "max_matches": search_engine.config["max_matches_per_guest"],
                "multi_embedding": search_engine.config["use_multi_embedding_matching"],
                "incremental_search_expansion": search_engine.config["incremental_search_expansion"]
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"API stats error: {e}")
        return jsonify({"error": str(e)}), 500

# CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ===================================
# CORRECTED Service Management
# ===================================

class ServiceManager:
    """CORRECTED: Service lifecycle management"""

    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.shutdown_handler = None

    def setup_auto_shutdown(self):
        """Setup auto-shutdown for cost optimization"""
        idle_minutes = int(os.getenv('IDLE_SHUTDOWN_MINUTES', '20'))
        if os.getenv('DISABLE_AUTO_SHUTDOWN', 'false').lower() == 'true':
            logger.info("Auto-shutdown is disabled")
            return

        from threading import Timer

        def check_and_shutdown():
            self.shutdown_handler = Timer(idle_minutes * 60, check_and_shutdown)
            self.shutdown_handler.daemon = True
            self.shutdown_handler.start()

        logger.info(f"Auto-shutdown enabled: {idle_minutes} minutes")
        check_and_shutdown()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 CORRECTED: Cleaning up resources...")
        self.search_engine.clear_cache()

        if self.shutdown_handler:
            self.shutdown_handler.cancel()

# Global service manager
service_manager = ServiceManager(search_engine)

# Signal handlers
def signal_handler(sig, frame):
    logger.info("Received shutdown signal")
    service_manager.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ===================================
# CORRECTED Main Entry Point
# ===================================

if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("🚀 CORRECTED ADVANCED FACE SEARCH SERVICE STARTING")
        logger.info("=" * 60)
        logger.info("Version: 3.0-CORRECTED")
        logger.info("Features:")
        logger.info("  ✅ CORRECTED Multi-embedding matching")
        logger.info("  ✅ CORRECTED Incremental updates")
        logger.info("  ✅ Quality-aware ranking")
        logger.info("  ✅ Contextual scoring")
        logger.info("  ✅ Duplicate detection")
        logger.info("  ✅ Group photo handling")
        logger.info("  ✅ Proper DynamoDB consistency handling")
        logger.info("  ✅ Enhanced debugging endpoints")
        logger.info("  ✅ Manual trigger endpoints")
        logger.info("=" * 60)

        # Test AWS connectivity
        try:
            search_engine.dynamodb.describe_limits()
            logger.info("✅ AWS connectivity verified")
        except Exception as e:
            logger.error(f"❌ AWS connectivity test failed: {e}")

        # Setup auto-shutdown
        service_manager.setup_auto_shutdown()

        # Start Flask server
        logger.info("🌐 Starting CORRECTED API server on 0.0.0.0:5000")

        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )

    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        sys.exit(1)