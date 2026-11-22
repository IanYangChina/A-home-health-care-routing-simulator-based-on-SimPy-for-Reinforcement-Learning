"use strict";

(() => {
  const T = globalThis.DispatchThree;
  const floorHeight = 2.8;
  const footprint = 13;
  const buildingKey = ([x, y]) => `${x},${y}`;
  const floorKey = point => point.join(",");
  const clamp = value => Math.max(0, Math.min(1, value));

  function route(leg) {
    const {from, to} = leg;
    const horizontal = Math.hypot(from[0] - to[0], from[1] - to[1]);
    return horizontal < 1e-12
      ? {points: [from, to], distances: [9.6 * Math.abs(from[2] - to[2])]}
      : {points: [from, [from[0], from[1], 0], [to[0], to[1], 0], to], distances: [from[2], horizontal, to[2]]};
  }

  function label(text) {
    const canvas = document.createElement("canvas");
    canvas.width = 112;
    canvas.height = 40;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#f5fcfa";
    ctx.fillRect(1, 1, 110, 38);
    ctx.strokeStyle = "#86aba7";
    ctx.strokeRect(1, 1, 110, 38);
    ctx.fillStyle = "#204848";
    ctx.font = "bold 25px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 56, 20);
    const sprite = new T.Sprite(new T.SpriteMaterial({map: new T.CanvasTexture(canvas), depthWrite: false}));
    sprite.scale.set(19.6, 7, 1);
    return sprite;
  }

  class DispatchMap3D {
    constructor(canvas, status, clock, fallback, resetButton) {
      this.canvas = canvas;
      this.status = status;
      this.clock = clock;
      this.motion = window.matchMedia("(prefers-reduced-motion: reduce)");
      try {
        const context = T && canvas.getContext("webgl2", {antialias: true, alpha: false});
        if (context) this.renderer = new T.WebGLRenderer({canvas, context, antialias: true});
      } catch {
        // The assignment interface also works on devices without WebGL.
      }
      if (!this.renderer) {
        canvas.hidden = clock.hidden = resetButton.hidden = true;
        document.getElementById("map-note").hidden = true;
        fallback.hidden = false;
        return;
      }

      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
      this.scene = new T.Scene();
      this.scene.background = new T.Color(0xeaf4f3);
      this.scene.add(new T.HemisphereLight(0xffffff, 0x9eb5a9, 2.0));
      const light = new T.DirectionalLight(0xffffff, 1.45);
      light.position.set(80, 145, 55);
      this.scene.add(light);
      this.camera = new T.OrthographicCamera(-200, 200, 180, -180, 1, 1200);
      this.controls = new T.OrbitControls(this.camera, canvas);
      this.controls.minZoom = 0.65;
      this.controls.maxZoom = 5;
      this.controls.maxPolarAngle = Math.PI * 0.48;
      this.controls.listenToKeyEvents(canvas);
      this.controls.addEventListener("change", () => this.schedule());
      resetButton.addEventListener("click", () => this.resetView());
      this.raycaster = new T.Raycaster();
      canvas.addEventListener("pointerdown", event => {
        if (event.isPrimary) this.pointer = {id: event.pointerId, x: event.clientX, y: event.clientY};
        else this.pointer = null;
      });
      canvas.addEventListener("pointerup", event => {
        const start = this.pointer;
        this.pointer = null;
        if (start?.id === event.pointerId && event.button === 0 &&
            Math.hypot(event.clientX - start.x, event.clientY - start.y) < 6) this.inspect(event);
      });
      const resizeObserver = new ResizeObserver(() => {
        const {clientWidth: width, clientHeight: height} = canvas;
        if (!width || !height) return;
        const aspect = width / height;
        const halfHeight = Math.max(155, 187 / aspect);
        Object.assign(this.camera, {left: -halfHeight * aspect, right: halfHeight * aspect,
          top: halfHeight, bottom: -halfHeight});
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height, false);
        this.schedule();
      });
      resizeObserver.observe(canvas);
    }

    resetView() {
      if (!this.renderer) return;
      this.camera.position.set(285, 232, 245);
      this.camera.zoom = 1;
      this.camera.updateProjectionMatrix();
      this.controls.target.set(0, 15, 0);
      this.controls.update();
      this.schedule();
    }

    world([x, y, z], lift = 0) {
      return new T.Vector3(x - this.center[0], z * floorHeight + lift, this.center[1] - y);
    }

    location(point) {
      const building = this.buildings.get(buildingKey(point));
      return building ? `${building.name} · Level ${point[2]}` : "Depot";
    }

    reset(simulator) {
      this.simulator = simulator;
      this.buildings = new Map();
      this.floors = new Map();
      const jobs = [...simulator.jobs].sort((a, b) => a.position[1] - b.position[1] || a.position[0] - b.position[0]);
      for (const job of jobs) {
        const key = buildingKey(job.position);
        if (!this.buildings.has(key)) {
          const id = String(this.buildings.size + 1).padStart(2, "0");
          this.buildings.set(key, {name: `Building ${id}`, badge: `B${id}`, floors: []});
        }
        const key3d = floorKey(job.position);
        if (!this.floors.has(key3d)) {
          const floor = {position: job.position, eligible: 0, total: 0};
          this.floors.set(key3d, floor);
          this.buildings.get(key).floors.push(floor);
        }
        this.floors.get(key3d).total++;
      }
      this.center = simulator.scenario.depot;
      this.inspecting = null;
      this.selected = null;
      this.preview = null;
      this.timeline = {from: 0, to: 0, started: 0, duration: 0};
      this.status.textContent = `${this.buildings.size} buildings · ${this.floors.size} occupied levels`;
      if (!this.renderer) return;

      if (this.site) {
        this.scene.remove(this.site);
        const resources = new Set([this.completedMaterial]);
        this.site.traverse(node => {
          if (node.geometry) resources.add(node.geometry);
          if (node.material) resources.add(node.material);
          if (node.material?.map) resources.add(node.material.map);
        });
        for (const item of resources) item.dispose();
      }
      this.site = new T.Group();
      this.scene.add(this.site);
      this.floorMarkers = [];
      const ground = new T.Mesh(new T.BoxGeometry(262, .55, 262), new T.MeshLambertMaterial({color: 0xc6dfd6}));
      ground.position.y = -.36;
      const grid = new T.GridHelper(250, 10, 0x9ab9b0, 0xaec9c0);
      this.site.add(ground, grid);
      const pinGeometry = new T.SphereGeometry(1.62, 10, 8);
      const shellMaterial = new T.MeshLambertMaterial({color: 0x5f9da6, transparent: true, opacity: .24, depthWrite: false});
      const roofMaterial = new T.MeshLambertMaterial({color: 0xa1c5c5});
      const edgeMaterial = new T.LineBasicMaterial({color: 0x648f94, transparent: true, opacity: .47});

      for (const building of this.buildings.values()) {
        const [x, y] = building.floors[0].position;
        const highest = Math.max(...building.floors.map(floor => floor.position[2]));
        const height = (highest + .46) * floorHeight;
        const shell = new T.BoxGeometry(footprint, height, footprint);
        const volume = new T.Mesh(shell, shellMaterial);
        volume.position.copy(this.world([x, y, 0], height / 2));
        const outline = new T.LineSegments(new T.EdgesGeometry(shell), edgeMaterial);
        outline.position.copy(volume.position);
        const roof = new T.Mesh(new T.BoxGeometry(footprint, .36, footprint), roofMaterial);
        roof.position.copy(this.world([x, y, 0], height));
        const badge = label(building.badge);
        badge.position.copy(this.world([x, y, 0], height + 6));
        this.site.add(volume, outline, roof, badge);
        const edges = [];
        const half = footprint / 2;
        for (let level = 1; level <= highest; level++) {
          const corners = [[-half, -half], [half, -half], [half, half], [-half, half]];
          corners.forEach(([dx, dz], i) => {
            const [ex, ez] = corners[(i + 1) % corners.length];
            edges.push(this.world([x + dx, y + dz, level]), this.world([x + ex, y + ez, level]));
          });
        }
        this.site.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(edges), edgeMaterial));
        for (const floor of building.floors) {
          const pin = new T.Mesh(pinGeometry, new T.MeshBasicMaterial({color: 0x779a9e}));
          pin.position.copy(this.world(floor.position, .15));
          pin.position.x += half + .95;
          pin.userData.floor = floor;
          floor.pin = pin;
          this.floorMarkers.push(pin);
          this.site.add(pin);
        }
      }

      const depot = new T.Mesh(new T.BoxGeometry(20, 3, 17), new T.MeshLambertMaterial({color: 0x34756c}));
      depot.position.copy(this.world(simulator.scenario.depot, 1.5));
      const depotLabel = label("DEPOT");
      depotLabel.position.copy(this.world(simulator.scenario.depot, 7.5));
      this.site.add(depot, depotLabel);
      this.destination = new T.Mesh(new T.BoxGeometry(footprint + 1, .43, footprint + 1),
        new T.MeshBasicMaterial({color: 0xa04ad4, transparent: true, opacity: .56, depthWrite: false}));
      this.destination.visible = false;
      this.routeLine = new T.Line(new T.BufferGeometry(), new T.LineBasicMaterial({color: 0x8f37be}));
      this.routeLine.visible = false;
      this.completedRoutes = new T.Group();
      this.completedMaterial = new T.LineBasicMaterial({color: 0x426e74, transparent: true, opacity: .68});
      const figure = new T.CapsuleGeometry(1.45, 2.35, 4, 8);
      this.previewMarker = new T.Mesh(figure, new T.MeshBasicMaterial({color: 0x9039bf, transparent: true, opacity: .67}));
      this.previewMarker.visible = false;
      this.nurseMarkers = simulator.nurses.map(() => {
        const marker = new T.Mesh(figure, new T.MeshLambertMaterial({color: 0x176d76}));
        this.site.add(marker);
        return marker;
      });
      this.site.add(this.destination, this.routeLine, this.completedRoutes, this.previewMarker);
      this.resetView();
    }

    select(floor) {
      const previous = this.selected;
      this.selected = floor;
      if (this.renderer && floor) {
        this.destination.visible = true;
        this.destination.position.copy(this.world(floor.position, .07));
        if (previous) this.styleFloor(previous);
        this.styleFloor(floor);
      }
    }

    styleFloor(floor) {
      const selected = floor === this.selected;
      floor.pin.material.color.setHex(selected ? 0x9735cf : floor.eligible ? 0x008579 : 0x729396);
      floor.pin.scale.setScalar(selected ? 1.6 : floor.eligible ? 1.19 : .77);
    }

    inspect(event) {
      const rect = this.canvas.getBoundingClientRect();
      const pointer = new T.Vector2(2 * (event.clientX - rect.left) / rect.width - 1,
        1 - 2 * (event.clientY - rect.top) / rect.height);
      this.raycaster.setFromCamera(pointer, this.camera);
      const hit = this.raycaster.intersectObjects(this.floorMarkers, false)[0];
      if (!hit) return;
      this.inspecting = hit.object.userData.floor;
      this.refreshInspection();
    }

    refreshInspection() {
      const {position, eligible, total} = this.inspecting;
      this.status.textContent = `${this.location(position)} · ${eligible} eligible of ${total} ${total === 1 ? "visit" : "visits"}`;
    }

    timeAt(now) {
      const {from, to, started, duration} = this.timeline;
      return duration ? from + (to - from) * clamp((now - started) / duration) : to;
    }

    update(rows, assignment) {
      for (const floor of this.floors.values()) floor.eligible = 0;
      for (const {job, forecast} of rows) {
        if (!forecast.reason) this.floors.get(floorKey(job.position)).eligible++;
      }
      if (assignment) {
        const {nurse, job} = assignment;
        this.inspecting = null;
        this.select(this.floors.get(floorKey(job.position)));
        this.status.textContent = `Nurse ${nurse.id + 1} assigned to visit #${job.id} · ${this.location(job.position)}`;
      } else if (this.inspecting) this.refreshInspection();
      if (!this.renderer) return;

      const now = performance.now();
      const {simulator} = this;
      if (simulator.time !== this.timeline.to) {
        const from = this.timeAt(now);
        this.timeline = {from, to: simulator.time, started: now,
          duration: this.motion.matches ? 0 : Math.min(1800, 430 + 135 * Math.max(0, simulator.time - from))};
        this.preview = null;
      }
      if (assignment) {
        const {leg} = assignment;
        this.routeLine.geometry.dispose();
        this.routeLine.geometry = new T.BufferGeometry().setFromPoints(route(leg).points.map(point => this.world(point, 1.25)));
        this.routeLine.visible = true;
        this.preview = !this.motion.matches && simulator.time <= leg.start && leg.end > leg.start
          ? {leg, started: now, duration: Math.max(650, Math.min(1800, (leg.end - leg.start) * 220))}
          : null;
      }
      for (const floor of this.floors.values()) this.styleFloor(floor);
      for (const child of this.completedRoutes.children) child.geometry.dispose();
      this.completedRoutes.clear();
      for (const nurse of simulator.done ? simulator.nurses : [simulator.current].filter(Boolean)) {
        const points = nurse.legs.filter(leg => leg.end <= simulator.time + 1e-8)
          .flatMap(leg => route(leg).points.map(point => this.world(point, .85)));
        if (points.length) this.completedRoutes.add(new T.Line(new T.BufferGeometry().setFromPoints(points), this.completedMaterial));
      }
      this.schedule();
    }

    pointOn(leg, fraction) {
      const {points, distances} = route(leg);
      let remaining = clamp(fraction) * distances.reduce((sum, distance) => sum + distance, 0);
      for (let i = 0; i < distances.length; i++) {
        const distance = distances[i];
        if (distance > 0 && remaining < distance) {
          return this.world(points[i], 2.7).lerp(this.world(points[i + 1], 2.7), remaining / distance);
        }
        remaining -= distance;
      }
      return this.world(leg.to, 2.7);
    }

    schedule() {
      if (!this.pendingFrame && this.site) this.pendingFrame = requestAnimationFrame(now => this.draw(now));
    }

    draw(now) {
      this.pendingFrame = null;
      const time = this.timeAt(now);
      for (const nurse of this.simulator.nurses) {
        const marker = this.nurseMarkers[nurse.id];
        const leg = nurse.legs.findLast(item => item.start <= time + 1e-8);
        if (leg && time < leg.end) {
          marker.position.copy(this.pointOn(leg, (time - leg.start) / (leg.end - leg.start)));
        } else {
          const stop = leg?.to ?? this.simulator.scenario.depot;
          marker.position.copy(this.world(stop, 2.7));
          if (stop === this.simulator.scenario.depot) {
            const angle = 2 * Math.PI * nurse.id / this.simulator.nurses.length;
            marker.position.x += 8 * Math.cos(angle);
            marker.position.z += 8 * Math.sin(angle);
            marker.position.y += 2;
          }
        }
        marker.material.color.setHex(nurse === this.simulator.current ? 0x124d97 : 0x12717c);
      }
      if (this.preview && now >= this.preview.started + this.preview.duration) this.preview = null;
      this.previewMarker.visible = Boolean(this.preview);
      if (this.preview) {
        const {leg, started, duration} = this.preview;
        this.previewMarker.position.copy(this.pointOn(leg, (now - started) / duration));
      }
      this.clock.textContent = `${this.preview ? "Route preview · " : ""}Minute ${time.toFixed(1)}`;
      this.renderer.render(this.scene, this.camera);
      if (this.preview || now < this.timeline.started + this.timeline.duration) this.schedule();
    }
  }

  globalThis.DispatchMap3D = DispatchMap3D;
})();
